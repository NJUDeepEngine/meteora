# coding=utf-8
# Copyright 2023-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import random
import math
import warnings
from contextlib import nullcontext
from typing import Any, List, Optional, Union, Tuple
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from transformers.pytorch_utils import Conv1D


from peft.tuners.lora.layer import Conv2d, Embedding
from peft.tuners.lora.config import LoraConfig
from peft.utils.other import transpose

from .tuners_utils import BaseTunerLayer, check_adapters_to_merge
from .layer_ops import moelinear_fwd_inner_bmm_torch, moelinear_fwd_inner_bmm_triton



# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class MoELoraLayer(BaseTunerLayer):
    # All names of layers that may contain (trainable) adapter weights
    adapter_layer_names = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B")
    # All names of other parameters that may contain adapter-related parameters
    other_param_names = ("r", "lora_alpha", "scaling", "lora_dropout")

    def __init__(self, base_layer: nn.Module, **kwargs) -> None:
        self.base_layer = base_layer
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_index = {}
        self.rmoe = 224
        # for key in self.r.keys:
        #     self.rmoe += self.r[key]
        in_features = self.get_base_layer().in_features
        out_features = self.get_base_layer().out_features

        self.loras = 0
        self.top_k = 2
        self.T = 1
        # gating
        self.moe_gate = nn.Linear(in_features, 28, bias=False)

        self.lora_dropout = nn.ModuleDict({}) # todo: dict to list?
        self.lora_A = nn.Linear(in_features, self.rmoe, bias=False)
        self.lora_B = nn.Linear(self.rmoe, out_features, bias=False)
        # For Embedding layer
        self.lora_embedding_A = nn.ParameterDict({})
        self.lora_embedding_B = nn.ParameterDict({})
        # Mark the weight as unmerged
        self._disable_adapters = False
        self.merged_adapters = []
        self.kwargs = kwargs

        base_layer = self.get_base_layer()
        if isinstance(base_layer, nn.Linear):
            in_features, out_features = base_layer.in_features, base_layer.out_features
        elif isinstance(base_layer, nn.Conv2d):
            in_features, out_features = base_layer.in_channels, base_layer.out_channels
        elif isinstance(base_layer, nn.Embedding):
            in_features, out_features = base_layer.num_embeddings, base_layer.embedding_dim
        elif isinstance(base_layer, Conv1D):
            in_features, out_features = (
                base_layer.weight.ds_shape if hasattr(base_layer.weight, "ds_shape") else base_layer.weight.shape
            )
        elif hasattr(base_layer, "infeatures") and hasattr(base_layer, "outfeatures"):
            # QuantLinear
            in_features, out_features = base_layer.infeatures, base_layer.outfeatures
        elif hasattr(base_layer, "input_size") and hasattr(base_layer, "output_size"):
            # Megatron ColumnParallelLinear,RowParallelLinear
            in_features, out_features = base_layer.input_size, base_layer.output_size
        else:
            raise ValueError(f"Unsupported layer type {type(base_layer)}")

        self.in_features = in_features
        self.out_features = out_features

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
        if r <= 0:
            raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
        # r = r
        # lora_alpha = 16
        lora_dropout = 0.1
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.rmoe = 0

        self.loras = len(self.r.keys())
        n_loras = len(self.r.keys())

        # gating


        for key in self.r.keys():
            self.rmoe += self.r[key]
        in_features = self.get_base_layer().in_features
        out_features = self.get_base_layer().out_features
        # self.moe_gate = nn.Linear(in_features, n_loras+1, bias=False)
        # self.moe_gate = nn.Linear(in_features, n_loras, bias=False)
        # self.lora_A = nn.Linear(in_features, self.rmoe, bias=False)
        # self.lora_B = nn.Linear(self.rmoe, out_features, bias=False)
        # print(adapter_name, self.r.keys(), self.lora_A, self.lora_B)
        # import copy
        # lora_A_new.weight = copy.deepcopy(self.lora_A.weight)
        # lora_B_new.weight = copy.deepcopy(self.lora_B.weight)

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            if adapter_name not in self.lora_index:
                lora_index = len(self.lora_index.keys())
                self.lora_index[adapter_name] = lora_index
                # self.lora_A[adapter_name] = nn.Linear(self.in_features, r, bias=False)
                # self.lora_B[adapter_name] = nn.Linear(r, self.out_features, bias=False)
            if use_rslora:
                self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
            else:
                self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights == "loftq":
            self.loftq_init(adapter_name)
        elif init_lora_weights:
            self.reset_lora_parameters(adapter_name, init_lora_weights)

        # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
                break
        self.set_adapter(self.active_adapters)

    def update_layer_post(self):
        in_features = self.get_base_layer().in_features
        out_features = self.get_base_layer().out_features
        n_loras = len(self.r.keys())
        self.moe_gate = nn.Linear(in_features, n_loras, bias=False)
        self.lora_A = nn.Linear(in_features, self.rmoe, bias=False)
        self.lora_B = nn.Linear(self.rmoe, out_features, bias=False)
         # check weight and qweight (for GPTQ)
        for weight_name in ("weight", "qweight"):
            weight = getattr(self.get_base_layer(), weight_name, None)
            if weight is not None:
                # the layer is already completely initialized, this is an update
                if weight.dtype.is_floating_point or weight.dtype.is_complex:
                    self.to(weight.device, dtype=weight.dtype)
                else:
                    self.to(weight.device)
                break

    def reset_lora_parameters(self, adapter_name, init_lora_weights):
        if init_lora_weights is False:
            return
        if init_lora_weights is True:
            # initialize A the same way as the default for nn.Linear and B to zero
            # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        elif init_lora_weights.lower() == "gaussian":
            nn.init.normal_(self.lora_A.weight, std=1 / self.r[adapter_name])
        else:
            raise ValueError(f"Unknown initialization {init_lora_weights=}")
        nn.init.zeros_(self.lora_B.weight)
        nn.init.normal_(self.moe_gate.weight)
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            nn.init.zeros_(self.lora_embedding_A[adapter_name])
            nn.init.normal_(self.lora_embedding_B[adapter_name])

    def loftq_init(self, adapter_name):
        from peft.utils.loftq_utils import loftq_init

        weight = self.get_base_layer().weight
        kwargs = {
            "num_bits": self.kwargs.get("loftq_bits", 4),
            "reduced_rank": self.r[adapter_name],
            "num_iter": self.kwargs.get("loftq_iter", 1),
        }

        qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            self.lora_A[adapter_name].weight.data = lora_A
            self.lora_B[adapter_name].weight.data = lora_B
        if adapter_name in self.lora_embedding_A.keys():
            # initialize a the same way as the default for nn.linear and b to zero
            self.lora_embedding_A[adapter_name].weight.data = lora_A
            self.lora_embedding_B[adapter_name].weight.data = lora_B
        self.get_base_layer().weight.data = qweight

    def set_scale(self, adapter, scale):
        if adapter not in self.scaling:
            # Ignore the case where the adapter is not in the layer
            return
        self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]

    def scale_layer(self, scale: float) -> None:
        if scale == 1:
            return

        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            self.scaling[active_adapter] *= scale

    def unscale_layer(self, scale=None) -> None:
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue

            if scale is None:
                self.scaling[active_adapter] = self.lora_alpha[active_adapter] / self.r[active_adapter]
            else:
                self.scaling[active_adapter] /= scale



# Below code is based on https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# and modified to work with PyTorch FSDP


#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------


class MoELinear(nn.Module, MoELoraLayer):
    # MoELora implemented in a dense layer
    def __init__(
        self,
        base_layer,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        MoELoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora)
        self.is_target_conv_1d_layer = is_target_conv_1d_layer

        # NOTE: add extra initialization for solving the speed problem for forward pass especially during inference
        self._extra_initialize()
        self._check_extra_initialize()

    def _extra_initialize(self):
        """do some extra initialization for MoELinear"""
        # the flag for whether the forward pass is prepared at once
        # it will be automatically set to True after the first forward
        self.fwd_prepared = False
        
        # to avoid unnecessary searching when mapping from idx to name
        self.lora_idx2name, self.lora_name2idx = None, None

        # this is views for self.lora_A and self.lora_B to treat each lora adapter seperatedly
        # i.e. if self.lora_A.weight.shape = (L*r, h), then self.lora_A_weights.shape = (L, r, h), where self.lora_A.weight[r*i:r*(i+1), :] == self.lora_A_weights[i]
        # note that: this requires all lora adapters to have the same rank r, so we use another flag to check this condition
        self.lora_A_weights, self.lora_B_weights, self.lora_A_mask = None, None, None
        self.all_ranks_are_identical, self.unique_rank = None, None
        self.scalings, self.unique_lora_dropout = None, None

        # NOTE: accelerate forward pass (speed up for 5x, i.e. only 2x slower than single lora now)
        # compared with the one using original mixtral MoE style (10x slower than single lora)
        self.use_accelerated_fwd = os.environ.get('MOELINEAR_USE_ACCELERATE_FWD', '1') == '1'
        
        # NOTE: when use accelerate forward, set the op implementation, 
        # choosing from {'torch', 'triton'}, default 'torch' for now
        self.accelerate_fwd_backend = os.environ.get('MOELINEAR_ACCELERATE_FWD_BACKEND', 'torch')
        
        self.accelerate_fwd_backend_torch_version = os.environ.get('MOELINEAR_ACCELERATE_FWD_BACKEND_TORCH_VERSION', 'v1')
        self.accelerate_fwd_backend_triton_version = os.environ.get('MOELINEAR_ACCELERATE_FWD_BACKEND_TRITON_VERSION', 'v4')
        self.accelerate_fwd_backend_triton_group_size = 16
        
        self.accelerate_fwd_inner_func = None # this unc is set by self.accelerate_fwd_backend in self._prepare_forward automatically

        # add this to apply normal lora
        # when there's only one moe adapter with index `gt_lora_idx`
        # (now only for debugging)
        self.single_moe_lora = os.environ.get('MOELINEAR_SINGLE_MOE_LORA', None) is not None
        self.gt_lora_idx = int(os.environ.get('MOELINEAR_SINGLE_MOE_LORA', '0'))

        ############    NOTE: choose `self.fwd_inner_loop_mode` from:
        # - 'all': 
        #   - to loop over each lora adapters and apply mm to selected sub-group tokens in each iteration
        # - 'parallel': 
        #   - to parallelly loop over each lora adapters and apply mm to selected sub-group tokens in each group of corresponding cuda.stream
        #   - now only for: 
        #       1. `self.training == False` if setting the flag: `self.fwd_inner_loop_pmode4train == False`, otherwise only for inference
        #   - FIXME: this mode is not even as quick as `self.fwd_inner_loop_mode == 'all'`
        # - 'batch': to apply selected lora adapters using bmm without looping over each lora adapter
        #   - now only for:
        #       1. `self.use_accelerated_fwd == True`
        #       2. the rank r is identical to all lora adapters, i.e. `self.unique_rank` (which is default in our framework settings right now)
        #       3. `self.training == False` if setting the flag: `self.fwd_inner_loop_bmode4train == False`, otherwise only for inference
        self.fwd_inner_loop_mode = os.environ.get('MOELINEAR_FWD_INNER_LOOP_MODE', 'batch')
        
        if self.fwd_inner_loop_mode == 'parallel':
            self.fwd_inner_loop_pmode4train = False # FIXME: can this mode work well in training? if so, toggle this flag to True
            self.fwd_inner_loop_psize = 8 # the maximum number of parallel streams
        elif self.fwd_inner_loop_mode == 'batch':
            self.fwd_inner_loop_bmode4train = False # FIXME: can this mode work well in training? if so, toggle this flag to True
        
        self.parallel_fwd_inner_loop, self.fwd_inner_loop_stream_contexts = None, None # a flag with a list of stream contexts, auto create in `self._prepare_forward`
        
    def _check_extra_initialize(self):
        assert self.fwd_inner_loop_mode in ["all", "parallel", "batch",]
        assert self.accelerate_fwd_backend in ["torch", "triton",]
        assert self.accelerate_fwd_backend_torch_version in ["v1", "v2"]
        assert self.accelerate_fwd_backend_triton_version in ["v1", "v2", "v3", "v4"]

    def merge(self, safe_merge: bool = False, adapter_names: Optional[List[str]] = None) -> None:
        """
        Merge the active adapter weights into the base weights

        Args:
            safe_merge (`bool`, *optional*):
                If True, the merge operation will be performed in a copy of the original weights and check for NaNs
                before merging the weights. This is useful if you want to check if the merge operation will produce
                NaNs. Defaults to `False`.
            adapter_names (`List[str]`, *optional*):
                The list of adapter names that should be merged. If None, all active adapters will be merged. Defaults
                to `None`.
        """
        if self.merged:
            warnings.warn(
                f"Already following adapters were merged {','.join(self.merged_adapters)}. "
                f"You are now additionally merging {','.join(self.active_adapters)}."
            )

        if adapter_names is None:
            adapter_names = self.active_adapters

        for active_adapter in adapter_names:
            if active_adapter in self.lora_A.keys():
                base_layer = self.get_base_layer()
                if safe_merge:
                    # Note that safe_merge will be slower than the normal merge
                    # because of the copy operation.
                    orig_weights = base_layer.weight.data.clone()
                    orig_weights += self.get_delta_weight(active_adapter)

                    if not torch.isfinite(orig_weights).all():
                        raise ValueError(
                            f"NaNs detected in the merged weights. The adapter {active_adapter} seems to be broken"
                        )

                    base_layer.weight.data = orig_weights
                else:
                    base_layer.weight.data += self.get_delta_weight(active_adapter)
                self.merged_adapters.append(active_adapter)

    def unmerge(self) -> None:
        """
        This method unmerges all merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return
        while len(self.merged_adapters) > 0:
            active_adapter = self.merged_adapters.pop()
            if active_adapter in self.lora_A.keys():
                self.get_base_layer().weight.data -= self.get_delta_weight(active_adapter)

    def get_delta_weight(self, adapter) -> torch.Tensor:
        """
        Compute the delta weight for the given adapter.

        Args:
            adapter (str):
                The name of the adapter for which the delta weight should be computed.
        """
        device = self.lora_B[adapter].weight.device
        dtype = self.lora_B[adapter].weight.dtype

        # In case users wants to merge the adapter weights that are in
        # float16 while being on CPU, we need to cast the weights to float32, perform the merge and then cast back to
        # float16 because the `@` and matmul operation in general is not supported in torch + cpu + fp16.
        cast_to_fp32 = device.type == "cpu" and dtype == torch.float16

        weight_A = self.lora_A[adapter].weight
        weight_B = self.lora_B[adapter].weight

        if cast_to_fp32:
            weight_A = weight_A.float()
            weight_B = weight_B.float()

        output_tensor = transpose(weight_B @ weight_A, self.fan_in_fan_out) * self.scaling[adapter]

        if cast_to_fp32:
            output_tensor = output_tensor.to(dtype=dtype)

            # cast back the weights
            self.lora_A[adapter].weight.data = weight_A.to(dtype)
            self.lora_B[adapter].weight.data = weight_B.to(dtype)

        return output_tensor

    def _prepare_forward(self):
        """prepare some variables before forward at once"""

        ## prepare the one2one mapping between lora_idx and lora_name
        if self.lora_name2idx is None:
            self.lora_name2idx = self.lora_index
        if self.lora_idx2name is None:
            self.lora_idx2name = {v:k for k, v in self.lora_name2idx.items()}

        ## prepare others beforing applying moe loras
        if self.use_accelerated_fwd and self.fwd_inner_loop_mode == "batch" \
            and (self.fwd_inner_loop_bmode4train or not self.training) and self.is_all_ranks_identical():
            if self.scalings is None:
                self.scalings = torch.zeros(self.loras, device=self.lora_A.weight.device, dtype=torch.float32) # shape = (l,)
                for i in range(self.loras): self.scalings[i] = self.scaling[self.lora_idx2name[i]]
            if self.unique_lora_dropout is None: # BUG: now just take the first dropout to use, since all p are the same for this framework now
                self.unique_lora_dropout = self.lora_dropout[self.lora_idx2name[0]]
            if self.lora_A_weights is None:
                self.lora_A_weights = self.lora_A.weight.view(self.loras, self.unique_rank, -1).transpose(1,2) # (l, h, r)
                if self.accelerate_fwd_backend == "triton":
                    h, m, r = self.lora_A_weights.shape[1], self.accelerate_fwd_backend_triton_group_size, self.unique_rank
                    self.lora_A_weights = torch.cat(self.lora_A_weights.split(h//m, dim=1), dim=2).contiguous() # shape from (l, h, r) to (l, h//m, r*m)
                    if self.accelerate_fwd_backend_triton_version == "v3":
                        self.lora_A_mask = torch.zeros(m, r*m, dtype=torch.bool, device=self.lora_A_weights.device) # shape: (m, r*m)
                        for i in range(m): self.lora_A_mask[i, i*r:(i+1)*r] = True
                    elif self.accelerate_fwd_backend_triton_version == "v4":
                        self.lora_A_weights *= self.scalings[: , None, None] # prescaling on lora_A
                        lora_A_mask1 = torch.zeros(m, r*m, dtype=self.lora_A_weights.dtype, device=self.lora_A_weights.device) # shape: (m, r*m)
                        for i in range(m): lora_A_mask1[i, i*r:(i+1)*r] = 1.
                        lora_A_mask2 = torch.cat([torch.eye(r, dtype=self.lora_A_weights.dtype, device=self.lora_A_weights.device) for _ in range(m)], dim=0).contiguous() # shape: (r*m, r)
                        self.lora_A_mask = (lora_A_mask1, lora_A_mask2) # mask olA by `olA * lA_mask1 @ lA_mask2` (shape: (m, r*m) => (m, r)) to be used in `olA @ lB`
            if self.lora_B_weights is None:
                self.lora_B_weights = self.lora_B.weight.view(-1, self.loras, self.unique_rank).permute(1,2,0).contiguous() # (l, r, hout)
            
        self.parallel_fwd_inner_loop = self.use_accelerated_fwd and self.fwd_inner_loop_mode == 'parallel' and (self.fwd_inner_loop_pmode4train or not self.training)
        if self.parallel_fwd_inner_loop:
            stream_pools = [torch.cuda.Stream() for _ in range(min(self.fwd_inner_loop_psize, self.loras))]
            self.fwd_inner_loop_stream_contexts = [torch.cuda.stream(stream_pools[i % self.fwd_inner_loop_psize]) for i in range(self.loras)]
        else:
            null_context = nullcontext()
            self.fwd_inner_loop_stream_contexts = [null_context for _ in range(self.loras)]
        
        
        if self.use_accelerated_fwd and self.accelerate_fwd_inner_func is None:
            self.accelerate_fwd_inner_func = {
                "torch": partial(
                            moelinear_fwd_inner_bmm_torch,
                            version=self.accelerate_fwd_backend_torch_version,
                        ),
                "triton": partial(
                            moelinear_fwd_inner_bmm_triton, 
                            version=self.accelerate_fwd_backend_triton_version
                        ),
            }[self.accelerate_fwd_backend]
        
        self.fwd_prepared = True
        
    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        ## preprare output placeholder
        prev_dtype = x.dtype
        moe_logits, moe_weights, selected_loras = None, None, None
        
        ## prepare forward
        if not self.fwd_prepared: self._prepare_forward()
        
        ## apply moe loras
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        elif self.single_moe_lora:
            result = self.base_layer(x, *args, **kwargs)
            lora_idx = self.gt_lora_idx if self.gt_lora_idx is not None else 0
            adapter_name = self.lora_idx2name[lora_idx]
            ith_adapter_r = self.r[adapter_name]
            start = lora_idx * ith_adapter_r
            end = (lora_idx + 1) * ith_adapter_r
            ith_lora_A_weight = self.lora_A.weight[start:end ,:]
            ith_lora_B_weight = self.lora_B.weight[:, start:end]
            dropout = self.lora_dropout[adapter_name]
            scaling = self.scaling[adapter_name]

            x = x.to(self.lora_A.weight.dtype)
            result = result + F.linear(F.linear(dropout(x), ith_lora_A_weight), ith_lora_B_weight) * scaling
        elif not self.use_accelerated_fwd:
            result = self.base_layer(x, *args, **kwargs)

            batch_size, sequence_length, hidden_dim = x.shape
            x = x.view(-1, hidden_dim)

            moe_logits = self.moe_gate(x)
            moe_weights = F.softmax(moe_logits, dim=1, dtype=torch.float)
            moe_weights, selected_loras = torch.topk(moe_weights, self.top_k, dim=-1)
            moe_weights /= moe_weights.sum(dim=-1, keepdim=True)

            # we cast back to the input dtype
            moe_weights = moe_weights.to(x.dtype)
            out_features = self.lora_B.out_features
            final_x = torch.zeros(
                (batch_size * sequence_length, out_features), dtype=x.dtype, device=x.device
            )
            # we cast back to the input dtype
            # One hot encode the selected experts to create an expert mask
            # this will be used to easily index which expert is going to be sollicitated
            # lora_mask = torch.nn.functional.one_hot(selected_loras, num_classes=(self.loras+1)).permute(2, 1, 0)
            lora_mask = torch.nn.functional.one_hot(selected_loras, num_classes=(self.loras)).permute(2, 1, 0)
            moe_logits = moe_logits.reshape(batch_size, sequence_length, self.loras)

            res1, res2 = self._original_forward_inner_loop(x, final_x, result, moe_logits, moe_weights, lora_mask, hidden_dim, prev_dtype)
            if res1 or res2: return res1, res2 # TODO: deal with non lora

            final_x = final_x.reshape(batch_size, sequence_length, out_features)
            moe_weights = moe_weights.reshape(batch_size, sequence_length, self.top_k)
            selected_loras = selected_loras.reshape(batch_size, sequence_length, self.top_k)
            result += final_x
        else:
            result = self.base_layer(x, *args, **kwargs)

            batch_size, sequence_length, hidden_dim = x.shape
            out_features = self.lora_B.out_features

            x = x.view(-1, hidden_dim)
            if self.training:
                moe_logits = self.moe_gate(x) # only need to return moe logits in training
                moe_weights, selected_loras = torch.topk(moe_logits, self.top_k, dim=-1)
            else: moe_weights, selected_loras = torch.topk(self.moe_gate(x), self.top_k, dim=-1)

            if self.top_k == 1: moe_weights[:] = 1. # when k=1, then weights must be 1.
            # else: moe_weights = F.softmax(moe_weights, dim=1, dtype=torch.float).to(x.dtype)
            else: moe_weights = F.softmax(moe_weights/self.T, dim=1, dtype=torch.float).to(x.dtype)

            x = x.to(self.lora_A.weight.dtype)
            result = result.view(-1, out_features)

            if self.fwd_inner_loop_mode == "batch" and \
                (self.fwd_inner_loop_bmode4train or not self.training) and \
                self.is_all_ranks_identical():
                self._accelerated_forward_inner_bmm(x, result, moe_weights, selected_loras)
            else:
                self._accelerated_forward_inner_loop(x, result, moe_weights, selected_loras)

            result = result.view(batch_size, sequence_length, out_features)

            if self.training:
                moe_weights = moe_weights.view(batch_size, sequence_length, self.top_k)
                moe_logits = moe_logits.view(batch_size, sequence_length, self.loras)
                selected_loras = selected_loras.view(batch_size, sequence_length, self.top_k)

        ## return output
        return result.to(prev_dtype), moe_logits

    def _accelerated_forward_inner_bmm(self, x, result, moe_weights, selected_loras):
        
        if self.accelerate_fwd_backend == "triton" and self.accelerate_fwd_backend_triton_version in ["v3", "v4"]:
            self.accelerate_fwd_inner_func(
                x=x, result=result, 
                lora_A_weights=self.lora_A_weights, lora_B_weights=self.lora_B_weights,
                scalings=self.scalings, lora_dropout=self.unique_lora_dropout,
                moe_weights=moe_weights, selected_loras=selected_loras,
                lora_A_mask=self.lora_A_mask,
            )
        else:
            self.accelerate_fwd_inner_func(
                x=x, result=result, 
                lora_A_weights=self.lora_A_weights, lora_B_weights=self.lora_B_weights,
                scalings=self.scalings, lora_dropout=self.unique_lora_dropout,
                moe_weights=moe_weights, selected_loras=selected_loras,
            )
            
    def _accelerated_forward_inner_loop(self, x, result, moe_weights, selected_loras):
        lora_mask = F.one_hot(selected_loras, num_classes=(self.loras))

        for lora_idx in range(self.loras):

            with self.fwd_inner_loop_stream_contexts[lora_idx]:
                token_idx, top_idx = torch.where(lora_mask[:, :, lora_idx])
                if len(token_idx) == 0: continue

                adapter_name = self.lora_idx2name[lora_idx]
                ith_adapter_r = self.r[adapter_name]

                start = lora_idx * ith_adapter_r
                end = (lora_idx + 1) * ith_adapter_r
                ith_lora_A_weight = self.lora_A.weight[start:end ,:]
                ith_lora_B_weight = self.lora_B.weight[:, start:end]

                lora_dropout = self.lora_dropout[adapter_name]
                scaling = self.scaling[adapter_name]

                ith_lora_result = F.linear(
                    F.linear(lora_dropout(x[token_idx, :]), ith_lora_A_weight), ith_lora_B_weight) \
                    * scaling * moe_weights[token_idx, top_idx, None]

                result.index_add_(0, token_idx, ith_lora_result.to(x.dtype))

        # Synchronize all streams to ensure all iterations are completed
        if self.parallel_fwd_inner_loop: torch.cuda.synchronize()

    def _original_forward_inner_loop(self, x, final_x, result, moe_logits, moe_weights, lora_mask,
                            hidden_dim, prev_dtype) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:

        for lora_idx in range(self.loras):
            adapter_name = None
            if lora_idx == self.loras: # TODO: deal with non lora
                result = result.to(prev_dtype)
                return result, moe_logits
            else:
                # adapter_name = list(self.lora_index.keys())[list(self.lora_index.values()).index(lora_idx)]
                adapter_name = self.lora_idx2name[lora_idx]

            ith_adapter_r = self.r[adapter_name]
            start = lora_idx * ith_adapter_r
            end = (lora_idx + 1) * ith_adapter_r

            ith_lora_A_weight = self.lora_A.weight[start:end ,:]
            ith_lora_B_weight = self.lora_B.weight[:, start:end]

            lora_dropout = self.lora_dropout[adapter_name]
            scaling = self.scaling[adapter_name]
            idx, top_x = torch.where(lora_mask[lora_idx])

            if top_x.shape[0] == 0:
                continue

            # in torch it is faster to index using lists than torch tensors
            top_x_list = top_x.tolist()
            idx_list = idx.tolist()

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            # current_state = x[None, top_x_list].reshape(-1, hidden_dim)
            # current_hidden_states = expert_layer(current_state) * routing_weights[top_x_list, idx_list, None]

            x = x.to(self.lora_A.weight.dtype)
            current_x = x[None, top_x_list].reshape(-1, hidden_dim)
            current_xA_T = F.linear(lora_dropout(current_x), ith_lora_A_weight, None)
            current_xA_TB_T = F.linear(current_xA_T, ith_lora_B_weight, None)
            # result += xA_TB_T * scaling * routing_weights[0]
            ith_lora_result = current_xA_TB_T * scaling
            # print(moe_weights.shape, moe_weights[top_x_list, idx_list, None])
            ith_lora_result = ith_lora_result * moe_weights[top_x_list, idx_list, None]
            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.

            final_x.index_add_(0, top_x, ith_lora_result.to(x.dtype))

        return None, None

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep

    def is_all_ranks_identical(self) -> bool:
        if self.all_ranks_are_identical is None:
            rs = set(self.r.values())
            self.all_ranks_are_identical = len(rs) == 1
            self.unique_rank = rs.pop()

        return self.all_ranks_are_identical


def dispatch_default_moe(
    target: torch.nn.Module,
    adapter_name: str,
    lora_config: LoraConfig,
    **kwargs,
) -> Optional[torch.nn.Module]:
    new_module = None

    if isinstance(target, BaseTunerLayer):
        target_base_layer = target.get_base_layer()
    else:
        target_base_layer = target

    if isinstance(target_base_layer, torch.nn.Embedding):
        embedding_kwargs = kwargs.copy()
        embedding_kwargs.pop("fan_in_fan_out", None)
        embedding_kwargs.update(lora_config.loftq_config)
        new_module = Embedding(target, adapter_name, **embedding_kwargs)
    elif isinstance(target_base_layer, torch.nn.Conv2d):
        kwargs.update(lora_config.loftq_config)
        new_module = Conv2d(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, torch.nn.Linear):
        if kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                "Setting fan_in_fan_out to False."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
        kwargs.update(lora_config.loftq_config)
        new_module = MoELinear(target, adapter_name, **kwargs)
    elif isinstance(target_base_layer, Conv1D):
        if not kwargs["fan_in_fan_out"]:
            warnings.warn(
                "fan_in_fan_out is set to False but the target module is `Conv1D`. " "Setting fan_in_fan_out to True."
            )
            kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
        kwargs.update(lora_config.loftq_config)
        new_module = MoELinear(target, adapter_name, is_target_conv_1d_layer=True, **kwargs)

    return new_module
