
from typing import Tuple

import torch
import torch.nn.functional as F

import triton

from .layer_ops_triton_kernel import triton_bmm, blora_fwd_kernel, blora_fwd_kernel_with_loraA_mask


def _get_strides(x, dims=None, ndims=-1):
    
    strides = [x.stride(i) for i in range(len(x.shape))]
    
    if dims is not None: return [s for i, s in enumerate(strides) if i in dims]
    if ndims is not None: return strides[:ndims]
    return strides


###########################        version 4        ###########################

# copied from torch v2, but reshape (x, blA, blB) to avoid too many unnecessary computations

def _moelinear_fwd_inner_bmm_triton_v4(
        x: torch.Tensor, result: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor, 
        lora_A_mask: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
    """the inner bmm process in MoELinear.forward implemented by triton
    Args:
        x: input tensor with shape (bs, h)
        result: output tensor with shape (bs, hout)
        lora_A_weights: lora A matrix with shape (l, h//m, r*m), and already pre-scaled by scalings!
        lora_B_weights: lora B matrix with shape (l, r, hout)
        scalings: scaling factor with shape (l,), do no need to use this since all lA are already pre-scaled !
        lora_dropout: lora dropout layer (during inference, it should equal to IdentityOp)
        moe_weights: moe weights with shape (bs, k)
        selected_loras: selected loras with shape (bs, k)
        lora_A_mask: (lA_mask1, lA_mask2), with shape: (m, r*m) | (r*m, r)
    Returns:
        result: updated output tensor with shape: (bs, hout) and dtype: x.dtype 
                (since the result is updated in-place, so the return value is not necessary)
    """
    
    # get shapes
    bs, k = x.shape[0], moe_weights.shape[-1]
    h, hout = x.shape[-1], result.shape[-1]
    hm, rm = lora_A_weights.shape[1:]
    r = lora_B_weights.shape[1]
    bsk, m = bs * k, h // hm
    
    # check shapes
    assert 16 <= rm <= 128 and 16 <= m <= 64 and 1 <= r <= 64, \
        f"""
        `rank (r) * group_size (m)` should be in the range [16, 128],
        `group_size (m)` should be in the range [16, 64],
        `rank (r)` should be in the range [1, 64],
        but we got: group_size: {m} | rank: {r} |
        """

    ##############################      preprocess input / output     ##############################
    
    ## prepare inputs
    x = x[:, None, :] # shape from (bs, h) to (bs, 1, h)
    x = lora_dropout(x).view(bs, m, hm) # shape from (bs, 1, h) => (bs, m, h//m)
    if k > 1: 
        moe_weights = moe_weights.view(-1)[:, None] # shape from (bs, k) => (bs*k, 1)
        selected_loras = selected_loras.view(-1) # shape from (bs, k) => (bs*k,)
    lA_mask1, lA_mask2 = lora_A_mask[0], lora_A_mask[1]; assert lA_mask1.shape == (m, rm) and lA_mask2.shape == (rm, r)

    ## prepare output buffer with shape: (bsk, hout)
    blora_results = torch.zeros((bsk, hout), dtype=x.dtype, device=x.device)
    
    ##############################      define meta dict / grid     ##############################
    ## define the meta dict including block sizes, num of wamps, num of stages, etc
    meta = dict(
        k=k, m=m, r=r, rm=rm, hm=hm, hout=hout,
        block_size_hout=128,
        block_size_hm=128,
        block_size_r=max(r, 16),
        num_warps=8,
        num_stages=4,
    )
    
    grid = lambda meta: (
        bsk,
        triton.cdiv(hout, meta["block_size_hout"]),
    )
    
    ##############################      launch the kernel     ##############################
    blora_fwd_kernel_with_loraA_mask[grid](
        x, *_get_strides(x), # shape = (bs, m, h//m), block on bs (bsk // k)
        blora_results, *_get_strides(blora_results), # shape = (bs*k, hout), block on bsk + hout
        lora_A_weights, *_get_strides(lora_A_weights), # shape = (l, h//m, r*m), no block, index on l
        lora_B_weights, *_get_strides(lora_B_weights), # shape = (l, r, hout), block on hout, index on l
        lA_mask1, *_get_strides(lA_mask1), # shape = (m, rm), no block
        lA_mask2, *_get_strides(lA_mask2), # shape = (rm, r), no block 
        selected_loras, # shape = (bsk, ), block on bsk
        **meta,
    )
    
    ##############################      postprocess output     ##############################
    ## pass scaling since the result is already pre-scaled on lora_A ! 
    ## apply moe weighted sum for topk loras
    if k > 1: 
        blora_results *= moe_weights # shape = (bs*k, hout)
        blora_results = blora_results.view(bs, k, -1).sum(dim=1) # shape from (bs*k, hout) to (bs, hout)
    
    ## add the blora result to the base result
    result.add_(blora_results)
    
    return result


###########################        version 3        ###########################

# copied from version 1, but reshape (x, blA, blB) to avoid too many unnecessary computations
def _moelinear_fwd_inner_bmm_triton_v3(
        x: torch.Tensor, result: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor, 
        lora_A_mask: torch.Tensor,
    ) -> torch.Tensor:
    """the inner bmm process in MoELinear.forward implemented by triton
    Args:
        x: input tensor with shape (bs, h)
        result: output tensor with shape (bs, hout)
        lora_A_weights: grouped lora A matrix with shape (l, h//m, r*m)
        lora_B_weights: lora B matrix with shape (l, r, hout)
        scalings: scaling factor with shape (l,)
        lora_dropout: lora dropout layer (during inference, it should equal to IdentityOp)
        moe_weights: moe weights with shape (bs, k)
        selected_loras: selected loras with shape (bs, k)
        lora_A_mask: lora A mask with shape (m, r*m)
    Returns:
        result: updated output tensor with shape (bs, hout) and x.dtype (since the result is updated in-place, so the return value is not necessary)
    """
    bs, k = x.shape[0], moe_weights.shape[-1]
    h, hout = x.shape[-1], result.shape[-1]
    m = h // lora_A_weights.shape[1]
    r = lora_A_weights.shape[-1] // m
    bsk = bs * k
    
    assert lora_A_mask.shape == (m, r*m)
    
    ## prepare batched input and moe weights
    x = x[:, None, :] # shape from (bs, h) to (bs, 1, h)
    if k > 1:
        x = x.expand(bs, k, -1).reshape(bs*k, 1, -1) # shape from (bs, 1, h) => (bs*k, 1, h)
        moe_weights = moe_weights.view(-1)[:, None] # shape from (bs, k) to (bs*k, 1)
    x = x.view(bs*k, m, h//m) # shape from (bs*k, 1, h) => (bs*k, m, h//m)

    ## prepare batched selected (loraA, loraB, scalings)
    selected_loras = selected_loras.view(-1) # shape from (bs, k) to (bs*k,)
    blora_A = lora_A_weights[selected_loras].contiguous() # shape from (l, h//m, r*m) to (bs*k, h//m, r*m)
    blora_B = lora_B_weights[selected_loras].contiguous() # shape from (l, r, hout) to (bs*k, r, hout)
    bscalings = scalings[selected_loras][:, None].contiguous() # from (l,) to (bs*k, 1)

    ## apply bmm for each (token_idx, top_idx) to its selected (loraA, loraB, scalings)
    blora_results = triton_bmm(
        triton_bmm(lora_dropout(x), blora_A)[lora_A_mask[None, :].expand(bsk, -1, -1)].view(bsk, m, -1).sum(dim=1, keepdim=True), # shape: (bs*k, m, h//m) => (bs*k, r*m) => (bs*k, m, r) => (bs*k, 1, r)
        blora_B
    ).squeeze(1) * bscalings
    
    ## apply moe weighted sum for topk loras
    if k > 1:
        blora_results *= moe_weights
        blora_results = blora_results.view(bs, k, -1).sum(dim=1) # shape from (bs*k, hout) to (bs, hout)

    ## add the blora result to the base result
    result.add_(blora_results.to(x.dtype))
    
    return result


###########################        version 2        ###########################

# copied from torch v1, and fuse torch.bmm(torch.bmm(x, blA), blB) into blora_fwd_kernel
# NOTE: here we also parallel batch dim into the maximum bsk blocks, 
# in which we only parallel dim hout by blocks
# and we also use selected loras to duplicately index (bs*k) loras without sharing
def _moelinear_fwd_inner_bmm_triton_v2(
        x: torch.Tensor, result: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor, 
    ) -> torch.Tensor:
    """the inner bmm process in MoELinear.forward implemented by triton
    Args:
        x: input tensor with shape (bs, h)
        result: output tensor with shape (bs, hout)
        lora_A_weights: lora A matrix with shape (l, h, r)
        lora_B_weights: lora B matrix with shape (l, r, hout)
        scalings: scaling factor with shape (l,)
        lora_dropout: lora dropout layer (during inference, it should equal to IdentityOp)
        moe_weights: moe weights with shape (bs, k)
        selected_loras: selected loras with shape (bs, k)
    Returns:
        result: updated output tensor with shape (bs, hout) and x.dtype (since the result is updated in-place, so the return value is not necessary)
    """
    
    bs, k = x.shape[0], moe_weights.shape[-1]
    r = lora_A_weights.shape[-1]
    h, hout = x.shape[-1], result.shape[-1]
    bsk = bs * k
    
    ##############################      preprocess input / output     ##############################
    x = x[:, None, :] # shape from (bs, h) to (bs, 1, h)
    ## prepare batched input and moe weights
    if k > 1:
        x = x.expand(bs, k, -1).reshape(bsk, 1, -1)  # shape from (bs, 1, h) to (bs*k, 1, h)
        moe_weights = moe_weights.view(-1)[:, None] # shape from (bs, k) to (bs*k, 1)
    x = lora_dropout(x) # shape: (bs*k, 1, h)
    
    ## prepare batched selected (loraA, loraB, scalings)
    selected_loras = selected_loras.view(-1) # shape from (bs, k) to (bs*k,)
    blora_A = lora_A_weights[selected_loras].contiguous() # shape from (l, h, r) to (bs*k, h, r)
    blora_B = lora_B_weights[selected_loras].contiguous() # shape from (l, r, hout) to (bs*k, r, hout)
    bscalings = scalings[selected_loras][:, None].contiguous() # from (l,) to (bs*k, 1)
    
    ## prepare output buffer (FIXME: should it be torch.float32 ?)
    blora_results = torch.zeros((bsk, 1, hout), dtype=torch.float32, device=x.device) # shape: (bs*k, 1, hout)
    
    ##############################      define meta dict / grid     ##############################
    ## define the meta dict including block sizes, num of wamps, num of stages, etc
    meta = dict(
        h=h, hout=hout, m=1, r=r, 
        block_size_h=128,
        block_size_hout=128,
        block_size_m=16, # NOTE: to avoid tl.dot error for all dimensions should be >= 16, we use the minimum size 16 here (actucally it is always 1)
        block_size_r=16,
        num_warps=4,
        num_stages=2,
    )
    
    grid = lambda meta: (
        bsk,
        triton.cdiv(hout, meta["block_size_hout"]),
    )
    
    ##############################      launch the kernel     ##############################
    blora_fwd_kernel[grid](
        x, x.stride(0), x.stride(1), x.stride(2),
        blora_results, blora_results.stride(0), blora_results.stride(1), blora_results.stride(2),
        blora_A, blora_A.stride(0), blora_A.stride(1), blora_A.stride(2),
        blora_B, blora_B.stride(0), blora_B.stride(1), blora_B.stride(2),
        **meta,
    )
    
    ##############################      postprocess output     ##############################
    ## scaling
    blora_results = blora_results.squeeze(1) * bscalings  # shape from (bs*k, 1, hout) to (bs*k, hout)
    ## apply moe weighted sum for topk loras
    if k > 1: 
        blora_results *= moe_weights # shape = (bs*k, hout)
        blora_results = blora_results.view(bs, k, -1).sum(dim=1) # shape from (bs*k, hout) to (bs, hout)
    
    ## add the blora result to the base result
    result.add_(blora_results.to(x.dtype))
    
    return result


###########################        version 1        ###########################

# copied from torch v1, and just replace torch.bmm for triton_bmm
def _moelinear_fwd_inner_bmm_triton_v1(
        x: torch.Tensor, result: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor, 
    ) -> torch.Tensor:
    """the inner bmm process in MoELinear.forward implemented by triton
    Args:
        x: input tensor with shape (bs, h)
        result: output tensor with shape (bs, hout)
        lora_A_weights: lora A matrix with shape (l, h, r)
        lora_B_weights: lora B matrix with shape (l, r, hout)
        scalings: scaling factor with shape (l,)
        lora_dropout: lora dropout layer (during inference, it should equal to IdentityOp)
        moe_weights: moe weights with shape (bs, k)
        selected_loras: selected loras with shape (bs, k)
    Returns:
        result: updated output tensor with shape (bs, hout) and x.dtype (since the result is updated in-place, so the return value is not necessary)
    """
    
    bs, k = x.shape[0], moe_weights.shape[-1]
    
    ## prepare batched input and moe weights
    x = x[:, None, :] # shape from (bs, h) to (bs, 1, h)
    if k > 1:
        x = x.expand(bs, k, -1).reshape(bs*k, 1, -1)  # shape from (bs, 1, h) to (bs*k, 1, h)
        moe_weights = moe_weights.view(-1)[:, None] # shape from (bs, k) to (bs*k, 1)

    ## prepare batched selected (loraA, loraB, scalings)
    selected_loras = selected_loras.view(-1) # shape from (bs, k) to (bs*k,)
    blora_A = lora_A_weights[selected_loras].contiguous() # shape from (l, h, r) to (bs*k, h, r)
    blora_B = lora_B_weights[selected_loras].contiguous() # shape from (l, r, hout) to (bs*k, r, hout)
    bscalings = scalings[selected_loras][:, None].contiguous() # from (l,) to (bs*k, 1)

    ## apply bmm for each (token_idx, top_idx) to its selected (loraA, loraB, scalings)
    blora_results = triton_bmm(
        triton_bmm(lora_dropout(x), blora_A), # shape: (bs*k, 1, h) => (bs*k, 1, r)
        blora_B # shape: (bs*k, 1, r) =>(bs*k, 1, hout)
    ).squeeze(1) * bscalings # shape: (bs*k, 1, hout) => (bs*k, hout)
    
    ## apply moe weighted sum for topk loras
    if k > 1:
        blora_results *= moe_weights
        blora_results = blora_results.view(bs, k, -1).sum(dim=1) # shape from (bs*k, hout) to (bs, hout)

    ## add the blora result to the base result
    result.add_(blora_results.to(x.dtype))
    
    return result


###########################        version 0        ###########################
# the trivial implementation to just copy the torch v1 one and add some noise
def _moelinear_fwd_inner_bmm_triton_v0(
        x: torch.Tensor, result: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor, 
    ) -> torch.Tensor:
    """the inner bmm process in MoELinear.forward implemented by triton
    Args:
        x: input tensor with shape (bs, h)
        result: output tensor with shape (bs, hout)
        lora_A_weights: lora A matrix with shape (l, h, r)
        lora_B_weights: lora B matrix with shape (l, r, hout)
        scalings: scaling factor with shape (l,)
        lora_dropout: lora dropout layer (during inference, it should equal to IdentityOp)
        moe_weights: moe weights with shape (bs, k)
        selected_loras: selected loras with shape (bs, k)
    Returns:
        result: updated output tensor with shape (bs, hout) and x.dtype (since the result is updated in-place, so the return value is not necessary)
    """
    from layer_ops_torch import _moelinear_fwd_inner_bmm_torch_v1
    
    _moelinear_fwd_inner_bmm_torch_v1(x, result, lora_A_weights, lora_B_weights, scalings, lora_dropout, moe_weights, selected_loras)
    
    return result + torch.randn_like(result) * 1e-3