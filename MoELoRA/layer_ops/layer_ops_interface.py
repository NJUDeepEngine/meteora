
from typing import List, Union

import torch

from .layer_ops_torch import (
    _moelinear_fwd_inner_bmm_torch_v1,
    _moelinear_fwd_inner_bmm_torch_v2,
)

from .layer_ops_triton import (
    _moelinear_fwd_inner_bmm_triton_v0,
    _moelinear_fwd_inner_bmm_triton_v1,
    _moelinear_fwd_inner_bmm_triton_v2,
    _moelinear_fwd_inner_bmm_triton_v3,
    _moelinear_fwd_inner_bmm_triton_v4,
)


def moelinear_fwd_inner_bmm_torch(
        x: torch.Tensor, result: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor,  
        version='v1',
    ) -> torch.Tensor:
    """the inner bmm process in MoELinear.forward implemented by torch
    Args:
        x: input tensor with shape (bs, h)
        result: output tensor with shape (bs, hout)
        lora_A_weights: lora A matrix with shape (l, h, r)
        lora_B_weights: lora B matrix with shape (l, r, hout)
        scalings: scaling factor with shape (l,)
        lora_dropout: lora dropout layer
        moe_weights: moe weights with shape (bs, k)
        selected_loras: selected loras with shape (bs, k)
        version: torch implementation version, default 'v1' for now
    Returns:
        result: updated output tensor with shape (bs, hout) and x.dtype (since the result is updated in-place, so the return value is not necessary)
    """
    
    kwargs = dict(
        x=x, result=result,
        lora_A_weights=lora_A_weights, lora_B_weights=lora_B_weights,
        scalings=scalings, lora_dropout=lora_dropout,
        moe_weights=moe_weights, selected_loras=selected_loras
    )
    
    if version == 'v1': return _moelinear_fwd_inner_bmm_torch_v1(**kwargs)
    elif version == 'v2': return _moelinear_fwd_inner_bmm_torch_v2(**kwargs)
    else: raise NotImplementedError(f"Unknown version {version} for moelinear_fwd_inner_bmm_torch")
    
    
def moelinear_fwd_inner_bmm_triton(
        x: torch.Tensor, result: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor, 
        lora_A_mask: Union[torch.Tensor, List[torch.Tensor]] = None,
        version='v4',
    ) -> torch.Tensor:
    """the inner bmm process in MoELinear.forward implemented by triton
    Args:
        x: input tensor with shape (bs, h)
        result: output tensor with shape (bs, hout)
        lora_A_weights: lora A matrix with shape (l, h, r)
        lora_B_weights: lora B matrix with shape (l, r, hout)
        scalings: scaling factor with shape (l,)
        lora_dropout: lora dropout layer
        moe_weights: moe weights with shape (bs, k)
        selected_loras: selected loras with shape (bs, k)
        lora_A_mask: lora A mask (s), this is only used in version >= v3
        version: the implementation version, default is 'v1'
    Returns:
        result: updated output tensor with shape (bs, hout) and x.dtype (since the result is updated in-place, so the return value is not necessary)
    """
    
    assert version in ['v1', 'v2', 'v3', 'v4']
    
    kwargs = dict(
        x=x, result=result,
        lora_A_weights=lora_A_weights, lora_B_weights=lora_B_weights,
        scalings=scalings, lora_dropout=lora_dropout,
        moe_weights=moe_weights, selected_loras=selected_loras
    )
    if version in ['v3', 'v4']: 
        assert lora_A_mask is not None
        kwargs['lora_A_mask'] = lora_A_mask
    
    if version == 'v0': return _moelinear_fwd_inner_bmm_triton_v0(**kwargs)
    elif version == 'v1': return _moelinear_fwd_inner_bmm_triton_v1(**kwargs)
    elif version == 'v2': return _moelinear_fwd_inner_bmm_triton_v2(**kwargs)
    elif version == 'v3': return _moelinear_fwd_inner_bmm_triton_v3(**kwargs)
    elif version == 'v4': return _moelinear_fwd_inner_bmm_triton_v4(**kwargs)
    else: raise NotImplementedError(f"Unknown version {version} for moelinear_fwd_inner_bmm_triton")