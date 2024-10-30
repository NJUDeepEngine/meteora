from typing import Tuple

import torch
import torch.nn.functional as F

import triton

from .layer_ops_triton_kernel_backward import triton_bmm, blora_bp_kernel, blora_bp_kernel_with_loraB_mask, blora_bp_kernel_without_mask, moe_weigths_bp_kernel


def _get_strides(x, dims=None, ndims=-1):
    
    strides = [x.stride(i) for i in range(len(x.shape))]
    
    if dims is not None: return [s for i, s in enumerate(strides) if i in dims]
    if ndims is not None: return strides[:ndims]
    return strides

###########################        version 4        ###########################

# copied from torch v2, but reshape (x, blA, blB) to avoid too many unnecessary computations

def _moelinear_bp_inner_bmm_triton_without_mask_v4(
        dr: torch.Tensor, dx: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor, 
        lora_B_mask: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
    
    # get shapes
    bs, k = dr.shape[0], moe_weights.shape[-1]
    h, hout = dx.shape[-1], dr.shape[-1]
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
    # x = x[:, None, :] # shape from (bs, h) to (bs, 1, h)
    # x = x.view(bs, hm, m) # shape from (bs, 1, h) => (bs, m, h//m)
    if k > 1: 
        moe_weights = moe_weights.view(-1)[:, None] # shape from (bs, k) => (bs*k, 1)
        selected_loras = selected_loras.view(-1) # shape from (bs, k) => (bs*k,)
    # lA_mask1, lA_mask2 = lora_B_mask[0], lora_B_mask[1]; assert lA_mask1.shape == (m, rm) and lA_mask2.shape == (rm, r)

    ## prepare output buffer with shape: (bsk, hout)
    blora_results = torch.zeros((bsk, hm, m), dtype=dr.dtype, device=dr.device)
    
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
        triton.cdiv(hm, meta["block_size_hm"]),
    )
    
    ##############################      launch the kernel     ##############################
    blora_bp_kernel_without_mask[grid](
        dr, *_get_strides(dr), # shape = (bs, hout), block on bs (bsk // k)
        blora_results, *_get_strides(blora_results), # shape = (bs*k, hm, m)
        lora_A_weights, *_get_strides(lora_A_weights), # shape = (l, h//m, r*m), no block, index on l
        lora_B_weights, *_get_strides(lora_B_weights), # shape = (l, r, hout), block on hout, index on l
        selected_loras, # shape = (bsk, ), block on bsk
        **meta,
    )
    
    ##############################      postprocess output     ##############################
    ## pass scaling since the result is already pre-scaled on lora_A ! 
    ## apply moe weighted sum for topk loras
    blora_results = blora_results.permute(0, 2, 1)
    blora_results = blora_results.view(bsk, -1)
    if k > 1: 
        blora_results *= moe_weights # shape = (bs*k, h)
        blora_results = blora_results.view(bs, k, -1).sum(dim=1) # shape from (bs*k, h) to (bs, h)
    
    ## add the blora result to the base result
    dx.add_(blora_results)
    
    return dx

def _moelinear_bp_inner_bmm_triton_v4(
        dr: torch.Tensor, dx: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor, 
        lora_B_mask: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
    
    # get shapes
    bs, k = dr.shape[0], moe_weights.shape[-1]
    h, hout = dx.shape[-1], dr.shape[-1]
    rn, hn = lora_B_weights.shape[1:]
    r = lora_A_weights.shape[2]
    bsk, n = bs * k, hout // hn
    
    # check shapes
    assert 16 <= rn <= 128 and 16 <= n <= 64 and 1 <= r <= 64, \
        f"""
        `rank (r) * group_size (m)` should be in the range [16, 128],
        `group_size (m)` should be in the range [16, 64],
        `rank (r)` should be in the range [1, 64],
        but we got: group_size: {n} | rank: {r} |
        """
    
    # prepare inputs
    dr = dr[:, None, :] # shape from (bs, hout) to (bs, 1, hout)
    dr = dr.view(bs, hn, n)
    if k > 1: 
        moe_weights = moe_weights.view(-1)[:, None] # shape from (bs, k) => (bs*k, 1)
        selected_loras = selected_loras.view(-1) # shape from (bs, k) => (bs*k,)
    lB_mask1, lB_mask2 = lora_B_mask[0], lora_B_mask[1]; assert lB_mask1.shape == (rn, n) and lB_mask2.shape == (r, rn)
    
    # prepare output buffer with shape (bsk, h)
    blora_results = torch.zeros((bsk, h), dtype=dr.dtype, device=dr.device)
    
    ##############################      define meta dict / grid     ##############################
    ## define the meta dict including block sizes, num of wamps, num of stages, etc
    meta = dict(
        k=k, n=n, r=r, rn=rn, hn=hn, h=h,
        block_size_h=128,
        block_size_hn=128,
        block_size_r=max(r, 16),
        num_warps=8,
        num_stages=4,
    )
    
    grid = lambda meta: (
        bsk,
        triton.cdiv(h, meta["block_size_h"]),
    )
    
    ##############################      launch the kernel     ##############################
    blora_bp_kernel_with_loraB_mask[grid](
        dr, *_get_strides(dr),
        blora_results, *_get_strides(blora_results), # shape = (bs*k, hout), block on bsk + hout
        lora_A_weights, *_get_strides(lora_A_weights), # shape = (l, h//m, r*m), no block, index on l
        lora_B_weights, *_get_strides(lora_B_weights), # shape = (l, r, hout), block on hout, index on l
        lB_mask1, *_get_strides(lB_mask1), # shape = (m, rm), no block
        lB_mask2, *_get_strides(lB_mask2), # shape = (rm, r), no block 
        selected_loras, # shape = (bsk, ), block on bsk
        **meta,
    )
    
    ##############################      postprocess output     ##############################
    ## pass scaling since the result is already pre-scaled on lora_A ! 
    ## apply moe weighted sum for topk loras
    if k > 1: 
        blora_results *= moe_weights # shape = (bs*k, hout)
        blora_results = blora_results.view(bs, k, -1).sum(dim=1) # shape from (bs*k, h) to (bs, h)
    
    # add the blora result to the base result
    dx.add_(blora_results)
    
    return dx


###########################        version 3        ###########################

# copied from version 1, but reshape (x, blA, blB) to avoid too many unnecessary computations
def _moelinear_bp_inner_bmm_triton_v3(
        dr: torch.Tensor, dx: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor, 
        lora_B_mask: torch.Tensor,
    ) -> torch.Tensor:
    bs, k = dr.shape[0], moe_weights.shape[-1]
    h, hout = dx.shape[-1], dr.shape[-1]
    n = hout // lora_B_weights.shape[-1]
    r = lora_B_weights.shape[1] // n
    bsk = bs * k
    
    assert lora_B_mask.shape == (n, r*n)
    
    # prepare batched input and moe weights
    dr = dr[:, None, :] # shape from (bs, hout) to (bs, 1, hout)
    if k > 1:
        dr = dr.expand(bs, k, -1).reshape(bs*k, 1, -1)  # shape from (bs, 1, h) to (bs*k, 1, h)
        moe_weights = moe_weights.view(-1)[:, None] # shape from (bs, k) to (bs*k, 1)
    dr = dr.view(bs*k, n, hout//n)
    
    # prepare batched selected (loraA, loraB, scalings)
    selected_loras = selected_loras.view(-1) # shape from (bs, k) to (bs*k,)
    blora_A = lora_A_weights[selected_loras].contiguous() # shape from (l, h, r) to (bs*k, h, r)
    blora_B = lora_B_weights[selected_loras].contiguous() # shape from (l, r*n, hout//n) to (bs*k, r*n, hout//n)
    blora_A = blora_A.permute(0, 2, 1).contiguous()
    blora_B = blora_B.permute(0, 2, 1).contiguous()
    bscalings = scalings[selected_loras][:, None].contiguous() # from (l,) to (bs*k, 1)
    
    # apply bmm for each (token_idx, top_idx) to its selected (loraA, loraB, scalings)
    blora_results = triton_bmm(
        triton_bmm(dr, blora_B)[lora_B_mask[None, :].expand(bsk, -1, -1)].view(bsk, n, -1).sum(dim=1, keepdim=True), # shape: (bs*k, n, h//n) => (bs*k, r*n) => (bs*k, n, r) => (bs*k, 1, r)
        blora_A # shape: (bs*k, 1, r) => (bs*k, 1, h)
    ).squeeze(1) * bscalings # shape: (bs*k, 1, h) => (bs*k, h)
    
    # apply moe weighted sum for topk loras
    if k > 1:
        blora_results *= moe_weights
        blora_results = blora_results.view(bs, k, -1).sum(dim=1) # shape from (bs*k, h) to (bs, h)
    
    # add the blora result to the base result
    dx.add_(blora_results.to(dr.dtype))
    
    return dx

###########################        version 2        ###########################

# copied from torch v1, and fuse torch.bmm(torch.bmm(x, blA), blB) into blora_fwd_kernel
# NOTE: here we also parallel batch dim into the maximum bsk blocks, 
# in which we only parallel dim hout by blocks
# and we also use selected loras to duplicately index (bs*k) loras without sharing
def _moelinear_bp_inner_bmm_triton_v2(
        dr: torch.Tensor, dx: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor, 
    ) -> torch.Tensor:
    bs, k = dr.shape[0], moe_weights.shape[-1]
    r = lora_A_weights.shape[-1]
    h, hout = dx.shape[-1], dr.shape[-1]
    bsk = bs * k
    
    ##############################      preprocess input / output     ##############################
    dr = dr[:, None, :] # shape from (bs, hout) to (bs, 1, hout)
    ## prepare batched input and moe weights
    if k > 1:
        dr = dr.expand(bs, k, -1).reshape(bsk, 1, -1)  # shape from (bs, 1, hout) to (bs*k, 1, hout)
        moe_weights = moe_weights.view(-1)[:, None] # shape from (bs, k) to (bs*k, 1)
    # x = lora_dropout(x) # shape: (bs*k, 1, h)
    
    ## prepare batched selected (loraA, loraB, scalings)
    selected_loras = selected_loras.view(-1) # shape from (bs, k) to (bs*k,)
    blora_A = lora_A_weights[selected_loras].contiguous() # shape from (l, h, r) to (bs*k, h, r)
    blora_B = lora_B_weights[selected_loras].contiguous() # shape from (l, r, hout) to (bs*k, r, hout)
    bscalings = scalings[selected_loras][:, None].contiguous() # from (l,) to (bs*k, 1)
    
    ## prepare output buffer (FIXME: should it be torch.float32 ?)
    blora_results = torch.zeros((bsk, 1, h), dtype=torch.float16, device=dr.device) # shape: (bs*k, 1, h)
    
    ##############################      define meta dict / grid     ##############################
    ## define the meta dict including block sizes, num of wamps, num of stages, etc
    meta = dict(
        h=h, hout=hout, m=1, r=r, 
        block_size_h=128,
        block_size_hout=128,
        block_size_m=16, # NOTE: to avoid tl.dot error for all dimensions should be >= 16, we use the minimum size 16 here (actucally it is always 1)
        block_size_r=max(16 ,r),
        num_warps=4,
        num_stages=2,
    )
    
    grid = lambda meta: (
        bsk,
        triton.cdiv(h, meta["block_size_h"]),
    )
    
    ##############################      launch the kernel     ##############################
    blora_bp_kernel[grid](
        dr, dr.stride(0), dr.stride(1), dr.stride(2),
        blora_results, blora_results.stride(0), blora_results.stride(1), blora_results.stride(2),
        blora_A, blora_A.stride(0), blora_A.stride(1), blora_A.stride(2),
        blora_B, blora_B.stride(0), blora_B.stride(1), blora_B.stride(2),
        **meta,
    )
    
    ##############################      postprocess output     ##############################
    ## scaling
    blora_results = blora_results.squeeze(1) * bscalings  # shape from (bs*k, 1, h) to (bs*k, h)
    ## apply moe weighted sum for topk loras
    if k > 1: 
        blora_results *= moe_weights # shape = (bs*k, h)
        blora_results = blora_results.view(bs, k, -1).sum(dim=1) # shape from (bs*k, hout) to (bs, h)
    
    ## add the blora result to the base result
    dx.add_(blora_results.to(dr.dtype))
    
    
    
    return dx

###########################        version 1        ###########################

# copied from torch v1, and just replace torch.bmm for triton_bmm
def _moelinear_bp_inner_bmm_triton_v1(
        dr: torch.Tensor, dx: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor, 
    ) -> torch.Tensor:
    
    bs, k = dr.shape[0], moe_weights.shape[-1]
    
    # prepare batched input and moe weights
    dr = dr[:, None, :] # shape from (bs, hout) to (bs, 1, hout)
    if k > 1:
        dr = dr.expand(bs, k, -1).reshape(bs*k, 1, -1)  # shape from (bs, 1, h) to (bs*k, 1, h)
        moe_weights = moe_weights.view(-1)[:, None] # shape from (bs, k) to (bs*k, 1)
    
    # prepare batched selected (loraA, loraB, scalings)
    selected_loras = selected_loras.view(-1) # shape from (bs, k) to (bs*k,)
    blora_A = lora_A_weights[selected_loras].contiguous() # shape from (l, h, r) to (bs*k, h, r)
    blora_B = lora_B_weights[selected_loras].contiguous() # shape from (l, r, hout) to (bs*k, r, hout)
    blora_A = blora_A.permute(0, 2, 1).contiguous()
    blora_B = blora_B.permute(0, 2, 1).contiguous()
    bscalings = scalings[selected_loras][:, None].contiguous() # from (l,) to (bs*k, 1)
    
    # apply bmm for each (token_idx, top_idx) to its selected (loraA, loraB, scalings)
    blora_results = triton_bmm(
        triton_bmm(dr, blora_B), # shape: (bs*k, 1, h) => (bs*k, 1, r)
        blora_A # shape: (bs*k, 1, r) => (bs*k, 1, h)
    ).squeeze(1) * bscalings # shape: (bs*k, 1, h) => (bs*k, h)
    
    # apply moe weighted sum for topk loras
    if k > 1:
        blora_results *= moe_weights
        blora_results = blora_results.view(bs, k, -1).sum(dim=1) # shape from (bs*k, h) to (bs, h)
    
    # add the blora result to the base result
    dx.add_(blora_results.to(dr.dtype))
    
    return dx


def _moelinear_bp_inner_bmm_triton_weights(
        dr: torch.Tensor, dk: torch.Tensor,
        dw: torch.Tensor, moe_weights: torch.Tensor
    ) -> torch.Tensor:
    bs, k = dr.shape[0], moe_weights.shape[-1]
    hout = dr.shape[-1]
    bsk = bs * k
    
    ##############################      preprocess input / output     ##############################
    if k > 1:
        dr = dr[:, None, :]
        dr = dr.expand(bs, k, -1)
        dr = dr.contiguous().view(bsk, -1) # shape = (bsk, hout)
        dk = dk.view(bsk, -1) # shape = (bsk, hout)
    
    # prepare output buffer
    weights_results = torch.zeros((bsk, hout), dtype=torch.float32, device=dr.device) # shape = (bsk, hout)
    
    ##############################      define meta dict / grid     ##############################
    ## define the meta dict including block sizes, num of wamps, num of stages, etc
    meta = dict(
        hout=hout,
        block_size_hout=128,
        num_warps=4,
        num_stages=2,
    )
    
    grid = lambda meta: (
        bsk,
        triton.cdiv(hout, meta["block_size_hout"])
    )
    
    ##############################      launch the kernel     ##############################
    moe_weigths_bp_kernel[grid](
        dr, dr.stride(0), dr.stride(1),
        dk, dk.stride(0), dk.stride(1),
        weights_results, weights_results.stride(0), weights_results.stride(1),
        **meta
    )
    
    ##############################      postprocess output     ##############################
    weights_results = weights_results.sum(dim=1).view(bs, k)
    weights_results /= moe_weights
    
    # add the weights result to the base result
    dw.add_(weights_results)
    
    return dw
    
    