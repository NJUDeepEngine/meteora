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

import math
import warnings
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D


from peft.tuners.lora.layer import Conv2d, Embedding

from peft.utils.other import transpose
from .tuners_utils import BaseTunerLayer, check_adapters_to_merge


from peft.tuners.lora.config import LoraConfig

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
        self.rmoe = 512
        # for key in self.r.keys:
        #     self.rmoe += self.r[key]
        in_features = self.get_base_layer().in_features
        out_features = self.get_base_layer().out_features
        
        self.loras = 0
        self.top_k = 1
        # gating
        self.gate = nn.Linear(in_features, 1, bias=False)
        
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
        self.gate = nn.Linear(in_features, n_loras+1, bias=False)
        self.lora_A = nn.Linear(in_features, self.rmoe, bias=False)
        self.lora_B = nn.Linear(self.rmoe, out_features, bias=False)
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

    # def update_layer_conv2d(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
    #     if r <= 0:
    #         raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
    #     self.r[adapter_name] = r
    #     self.lora_alpha[adapter_name] = lora_alpha
    #     if lora_dropout > 0.0:
    #         lora_dropout_layer = nn.Dropout(p=lora_dropout)
    #     else:
    #         lora_dropout_layer = nn.Identity()

    #     self.lora_dropout[adapter_name] = lora_dropout_layer
    #     # Actual trainable parameters
    #     base_layer = self.get_base_layer()
    #     if r > 0:
    #         kernel_size = base_layer.kernel_size
    #         stride = base_layer.stride
    #         padding = base_layer.padding
    #         self.lora_A[adapter_name] = nn.Conv2d(self.in_features, r, kernel_size, stride, padding, bias=False)
    #         self.lora_B[adapter_name] = nn.Conv2d(r, self.out_features, (1, 1), (1, 1), bias=False)
    #         if use_rslora:
    #             self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
    #         else:
    #             self.scaling[adapter_name] = lora_alpha / r

    #     if init_lora_weights == "loftq":
    #         self.loftq_init(adapter_name)
    #     elif init_lora_weights:
    #         self.reset_lora_parameters(adapter_name, init_lora_weights)

    #     weight = getattr(base_layer, "weight", None)
    #     if weight is not None:
    #         # the layer is already completely initialized, this is an update
    #         self.to(base_layer.weight.device, dtype=weight.dtype)
    #     self.set_adapter(self.active_adapters)

    # def update_layer_embedding(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora):
    #     if r <= 0:
    #         raise ValueError(f"`r` should be a positive integer value but the value passed is {r}")
    #     self.r[adapter_name] = r
    #     self.lora_alpha[adapter_name] = lora_alpha
    #     if lora_dropout > 0.0:
    #         lora_dropout_layer = nn.Dropout(p=lora_dropout)
    #     else:
    #         lora_dropout_layer = nn.Identity()

    #     self.lora_dropout[adapter_name] = lora_dropout_layer
    #     # Actual trainable parameters
    #     if r > 0:
    #         weight_A = torch.randn((r, self.in_features))
    #         weight_B = torch.randn((self.out_features, r))
    #         self.lora_embedding_A[adapter_name] = nn.Parameter(weight_A)
    #         self.lora_embedding_B[adapter_name] = nn.Parameter(weight_B)
    #         if use_rslora:
    #             self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
    #         else:
    #             self.scaling[adapter_name] = lora_alpha / r

    #     if init_lora_weights == "loftq":
    #         self.loftq_init(adapter_name)
    #     elif init_lora_weights:
    #         self.reset_lora_parameters(adapter_name, init_lora_weights)

    #     base_layer = self.get_base_layer()
    #     weight = getattr(base_layer, "weight", None)
    #     if weight is not None:
    #         # the layer is already completely initialized, this is an update
    #         self.to(base_layer.weight.device, dtype=weight.dtype)
    #     self.set_adapter(self.active_adapters)

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
        nn.init.normal_(self.gate.weight)
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

    def forward(self, x: torch.Tensor, adapter_name, *args: Any, **kwargs: Any) -> torch.Tensor:
        previous_dtype = x.dtype

        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            result = self.base_layer(x, *args, **kwargs)
        elif self.merged:
            result = self.base_layer(x, *args, **kwargs)
        else:
            result = self.base_layer(x, *args, **kwargs)
            
            
            # router_logits = self.gate(x)

            # routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            # routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
            # # routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            # # # we cast back to the input dtype
            # # routing_weights = routing_weights.to(x.dtype)
            # index = selected_experts[0]
            
            # if index == len(self.loras):
            #     result = self.base_layer(x, *args, **kwargs)
            # else:
                # adapter_name = list(self.lora_index.keys())[list(self.lora_index.values()).index(index)]
                
            index = self.lora_index[adapter_name]
            act_adapter_r = self.r[adapter_name]
            
            
            start = index * act_adapter_r
            end = (index + 1) * act_adapter_r
            lora_dropout = self.lora_dropout[adapter_name]
            scaling = self.scaling[adapter_name]
            x = x.to(self.lora_A.weight.dtype)
            xA_T = F.linear(lora_dropout(x), self.lora_A.weight[start:end ,:], None)
            xA_TB_T = F.linear(xA_T, self.lora_B.weight[:, start:end], None)
            # result += xA_TB_T * scaling * routing_weights[0]
            result += xA_TB_T * scaling
        result = result.to(previous_dtype)
        return result

    def __repr__(self) -> str:
        rep = super().__repr__()
        return "lora." + rep
    
    
    
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
