import torch
import torch.nn.functional as F

import triton
import triton.language as tl


@triton.jit
def bmm_kernel( # naive grid sequence organized in row-major order inter-group, and column-major order intra-group, with fp16 precision
    x_ptr, x_stride_b, x_stride_m, x_stride_k,
    y_ptr, y_stride_b, y_stride_k, y_stride_n,
    o_ptr, o_stride_b, o_stride_m, o_stride_n,
    m: tl.constexpr, n: tl.constexpr, k: tl.constexpr,
    # meta parameters
    block_size_m: tl.constexpr, block_size_n: tl.constexpr, block_size_k: tl.constexpr,
    group_size_m: tl.constexpr,
):
    ### init the pointers
    
    ## get the batch idx and (row,col) block idxs by groups in an inter-group row-major order and an intra-group col-major order 
    batch_idx = tl.program_id(axis=0) # the first axis for the batch idx in the batch
    block_idx = tl.program_id(axis=1) # the second axis for the block idx in the 2d matrix
    num_blocks_m = tl.cdiv(m, block_size_m) # the number of blocks in row dim
    num_blocks_n = tl.cdiv(n, block_size_n) # the number of blocks in col dim
    num_blocks_g = group_size_m * num_blocks_n # the number of 2d blocks in a group
    group_idx = block_idx // num_blocks_g # 1d group idx
    first_block_m = group_idx * group_size_m # the row block idx of the first block in that group
    group_size_m_ = min(num_blocks_m - first_block_m , group_size_m) # get the group size of that group, normally it equals `group_size_m`, but the last group may be less
    block_idx_m = first_block_m + (block_idx % group_size_m_) # get the row block idx of current block in that group in a col-major order
    block_idx_n = (block_idx % num_blocks_g) // group_size_m_ # get the col block idx of current block in that group in a col-major order
    
    ## get the 2d block pointers for o: shape=(block_size_m, block_size_n)
    block_ptrs_m = (block_idx_m * block_size_m + tl.arange(0, block_size_m)) % m # NOTE: mod m to avoid oob row offsets during computaion, but may leave redundant last row block idx: (m-1) and compute repeatitive values
    block_ptrs_n = (block_idx_n * block_size_n + tl.arange(0, block_size_n)) % n # NOTE: mod n to avoid oob col offsets during computaion, but may leave redundant last col block idx: (n-1) and compute repeatitive values
    
    ## get each 2d block pointers for x: shape=(block_size_m, block_size_k), y: shape=(block_size_k, block_size_n)
    offsets_k = tl.arange(0, block_size_k) # split inner dimk by blocks with size `block_size_k`
    x_block_ptrs = x_ptr + (block_ptrs_m[:, None] * x_stride_m + offsets_k[None, :] * x_stride_k) + (batch_idx * x_stride_b) # the start block along dimk for x
    y_block_ptrs = y_ptr + (offsets_k[:, None] * y_stride_k + block_ptrs_n[None, :] * y_stride_n) + (batch_idx * y_stride_b) # the start block along dimk for y
    
    
    ### do the operations
    # we iterate the kdim to compute a `[block_size_m, block_size_n]` block of the otuput matrix.
    # to get higher accuracy, we accumulate each block along dimk into a upcasted fp32 values `accumulator`
    # and the `accumulator` will be downcasted back to fp16 after the loop.
    o = tl.zeros((block_size_m, block_size_n), dtype=tl.float32) # upcast to fp32
    for block_idx_k in range(0, tl.cdiv(k, block_size_k)):
        block_offs_k = block_idx_k * block_size_k + offsets_k
        # load current block of x and y along dimk with mask to avoid oob, where oob values are set to 0.
        x = tl.load(x_block_ptrs, mask=block_offs_k[None, :] < k, other=0.)
        y = tl.load(y_block_ptrs, mask=block_offs_k[:, None] < k, other=0.)
        
        # compute x @ y for this block along dimk and add to `accumulator`
        o += tl.dot(x, y)
        
        # move x,y pointers to next block along dimk
        x_block_ptrs += block_size_k * x_stride_k
        y_block_ptrs += block_size_k * y_stride_k
    # o = o.to(tl.float16) # downcast to fl16
        
    ### store the output data from SRAM to HBM
    block_ptrs_m_ = block_idx_m * block_size_m + tl.arange(0, block_size_m)[:, None] # NOTE: here to store right output value, we need to remove the redundant (m-1) idx but use the real idx awith mask to avoid oob
    block_ptrs_n_ = block_idx_n * block_size_n + tl.arange(0, block_size_n)[None, :] # NOTE: here to store right output value, we need to remove the redundant (m-1) idx but use the real idx with mask to avoid oob
    o_ptrs = o_ptr + (block_ptrs_m_ * o_stride_m + block_ptrs_n_ * o_stride_n) + (batch_idx * o_stride_b)
    block_mask_mn = (block_ptrs_m_  < m) & (block_ptrs_n_ < n) # 2d block mask to avoid oob
    tl.store(o_ptrs, o, mask=block_mask_mn)
    

def triton_bmm(x: torch.tensor, y: torch.tensor) -> torch.tensor:
    # check the input
    assert x.dim() == y.dim() == 3
    assert x.shape[2] == y.shape[1] and x.dtype == y.dtype and x.device == y.device
    assert x.is_contiguous() and y.is_contiguous()
    
    # init the output buffer
    b,m,k = x.shape
    b,k,n = y.shape
    
    o = torch.zeros((b,m,n), dtype=x.dtype, device=x.device)
    
    # define some meta parameters
    # NOTE: we auto-tune the meta parameters like block_size and num_warps using triton.autotune decorator above the kernel
    meta = dict(
        block_size_m=16,
        block_size_n=128,
        block_size_k=16,
        group_size_m=8,
        num_warps=4,
        num_stages=4, 
    )
    
    # define the grid (number of blocks)
    grid = lambda meta: (
        b, 
        triton.cdiv(m, meta["block_size_m"]) * triton.cdiv(n, meta["block_size_n"]),
    ) # (b, m x n) blocks, each to compute matmul for xi block and yj block for the kth matrix pair in the batch
    
    # launch the kernel
    bmm_kernel[grid](
        x, x.stride(0), x.stride(1), x.stride(2),
        y, y.stride(0), y.stride(1), y.stride(2),
        o, o.stride(0), o.stride(1), o.stride(2),
        m, n, k,
        **meta,
    )
    
    return o 



@triton.jit
def blora_bp_kernel_without_mask(
        dr_ptr, dr_stride_bs,
        dx_ptr, dx_stride_bsk, dx_stride_hm,
        lA_ptr, lA_stride_l, lA_stride_hm,
        lB_ptr, lB_stride_l, lB_stride_r,
        sel_ptr, 
        k: tl.constexpr, m: tl.constexpr, r: tl.constexpr, rm: tl.constexpr, 
        hm: tl.constexpr, hout: tl.constexpr,
        block_size_hout: tl.constexpr, block_size_hm: tl.constexpr, block_size_r: tl.constexpr,
    ):
    
    ##################       get the block pointers        ##################
    # get the bsk block index and hout block index
    block_idx_bsk = tl.program_id(0)
    block_idx_bs = block_idx_bsk // k
    # block_idx_hout = tl.program_id(1)
    block_idx_hm = tl.program_id(1)
    
    # get block offsets
    offsets_hout = tl.arange(0, block_size_hout)
    offsets_hm = block_idx_hm * block_size_hm + tl.arange(0, block_size_hm)
    offsets_r = tl.arange(0, block_size_r)
    offsets_m = tl.arange(0, m)
    offsets_rm = tl.arange(0, rm)
    
    # get block masks
    block_mask_hout = offsets_hout < hout
    block_mask_hout_row = offsets_hout[None, :] < hout
    block_mask_hm_col = offsets_hm[:, None] < hm
    block_mask_rm_row = offsets_rm[None, :] < rm
    block_mask_r_col = offsets_r[:, None] < r
    block_mask_m_row = offsets_r[None, :] < m
    
    # get the selected lora idx
    sel_ptr += block_idx_bsk
    sel_idx = tl.load(sel_ptr)
    
    # get the block pointers for input variables: x, lA, lB, lA_mask1, lA_mask2
    lA_block_ptrs = lA_ptr + sel_idx * lA_stride_l + \
                (offsets_hm[:, None] * lA_stride_hm + offsets_rm[None, :]) # shape = (block_size_hm, rm)

    dr_block_ptrs = dr_ptr + block_idx_bs * dr_stride_bs + \
                ([[0]] + offsets_hout[None, :]) # shape = (1, block_size_hout)
    lA_block_ptrs = lA_ptr + sel_idx * lA_stride_l + \
                (offsets_hm[:, None] * lA_stride_hm + offsets_rm) # shape = (block_size_hm, rm)
    lB_block_ptrs = lB_ptr + sel_idx * lB_stride_l + \
                (offsets_r[:, None] * lB_stride_r + offsets_hout[None, :]) # shape = (block_size_r, block_size_hout)
    
    # get the block pointers for output variables, shape = (block_size_hm, m)
    dx_block_ptrs = dx_ptr + block_idx_bsk * dx_stride_bsk + offsets_hm[:, None] * dx_stride_hm + offsets_m[None, :]
    
    
    ##################      compute olB = lB.transpose @ dr      ##################
    compute_olB_dtype = tl.float16
    
    #init olB buffer, shape = (block_size_r, 1)
    olB = tl.zeros((block_size_r, 1))
    # loop over hout dim by blocks to accumulate olB
    for _ in range(tl.cdiv(hout, block_size_hout)):
        # update block masks
        block_mask_hout_col = offsets_hout[:, None] < hout
        block_mask_hout_row = offsets_hout[None, :] < hout
        
        # load dr, lB
        dr = tl.load(dr_block_ptrs, mask=block_mask_hout_row, other=0.).to(compute_olB_dtype) # shape = (1, block_size_hout)
        lB = tl.load(lB_block_ptrs, mask=block_mask_hout_row, others=0.).to(compute_olB_dtype) # shape = (block_size_r, block_size_hout)
        
        # compute olB = lB @ dr
        olB += tl.dot(lB, dr.T).to(compute_olB_dtype) # shape = (block_size_r, 1)
        
        # update block pointers and offsets
        offsets_hout += block_size_hout
        dr_block_ptrs += block_size_hout
        lB_block_ptrs += block_size_hout
    
    ##################            expand olB            ##################
    olB_r = torch.zeros(r*m, m, dtype=compute_olB_dtype, device = olB.device)
    for i in range(m): olB_r[i*r:(i+1)*r, i] = olB # shape = (r*m, m)
    
    ##################      compute olA = lA @ olB_r     ##################
    compute_olA_dtype = tl.float16
    
    # load lA
    lA = tl.load(lA_block_ptrs, mask = block_mask_hm_col & block_mask_rm_row, other=0.).to(compute_olA_dtype) # shape = (block_size_hm, rm)
    # compute olA = lA @ olB_r
    olA = tl.dot(lA, olB.to(compute_olA_dtype)) # shape = (block_size_hm, m)
    
    ##################           store output          ##################
    # store olA with shape = (block_size_hm, m)
    tl.store(dx_block_ptrs, olA.to(dx_block_ptrs.dtype.element_ty), mask=block_mask_hm_col & block_mask_m_row)

@triton.jit
def blora_bp_kernel_with_loraB_mask(
        dr_ptr, dr_stride_bs, dr_stride_n,
        dx_ptr, dx_stride_bsk,
        lA_ptr, lA_stride_l, lA_stride_h,
        lB_ptr, lB_stride_l, lB_stride_rn,
        lB_mask1_ptr, lB_mask1_stride_rn,
        lB_mask2_ptr, lB_mask2_stride_r,
        sel_ptr, 
        k: tl.constexpr, n: tl.constexpr,
        r: tl.constexpr, rn: tl.constexpr, 
        hn: tl.constexpr, h: tl.constexpr,
        block_size_r: tl.constexpr,
        block_size_hn: tl.constexpr, block_size_h: tl.constexpr,
    ):
    """ 
    dr shape = (bs, n, hout//n)
    lora_B_weights shape = (loras, r, hout) -> (loras, rn, hout//n)
    lora_A_weights shape = (loras, h, r)
    """
    
    ##################       get the block pointers        ##################
    # get the bsk block index and hout block index
    block_idx_bsk = tl.program_id(0)
    block_idx_bs = block_idx_bsk // k
    block_idx_h = tl.program_id(1)
    
    # get block offsets
    offsets_h = block_idx_h * block_size_h + tl.arange(0, block_size_h)
    offsets_hn = tl.arange(0, block_size_hn)
    offsets_r = tl.arange(0, block_size_r)
    offsets_n = tl.arange(0, n)
    offsets_rn = tl.arange(0, rn)
    
    # get block mask
    block_mask_h = offsets_h < h
    block_mask_h_col = offsets_h[:, None] < h
    block_mask_r_col = offsets_r[:, None] < r
    block_mask_r_row = offsets_r[None, :] < r
    
    # get the selected lora idx
    sel_ptr += block_idx_bsk
    sel_idx = tl.load(sel_ptr)
    
    # get the block pointers for input variables: dr, lA, lB, lB_mask1, lB_mask2
    dr_block_ptrs = dr_ptr + block_idx_bs * dr_stride_bs + \
                (offsets_n[:, None] * dr_stride_n + offsets_hn[None, :]) # shape = (n, block_size_hn)
    lB_block_ptrs = lB_ptr + sel_idx * lB_stride_l + \
                (offsets_rn[:, None] * lB_stride_rn + offsets_hn[None, :]) # shape = (rn, block_size_hn)
    lA_block_ptrs = lA_ptr + sel_idx * lA_stride_l + \
                (offsets_h[:, None] * lA_stride_h + offsets_r[None, :]) # shape = (block_size_h, block_size_r)
    lB_mask1_ptrs = lB_mask1_ptr + (offsets_rn[:, None] * lB_mask1_stride_rn + offsets_n[None, :]) # shape = (rn, n)
    lB_mask2_ptrs = lB_mask2_ptr + (offsets_r[:, None] * lB_mask2_stride_r + offsets_rn[None, :]) # shape = (block_size_r, rn)
    
    # get the block pointers for output variables
    dx_block_ptrs = dx_ptr + block_idx_bsk * dx_stride_bsk + offsets_h # shape = (block_size_h, )
    
    ##################      compute olB = lB @ dr.T      ##################
    compute_olB_dtype = tl.float16
    
    # init olB buffer
    olB = tl.zeros((rn, n), dtype=compute_olB_dtype) # shape = (rn, n)
    # loop over hn dim by blocks to accumulate olB
    for block_idx_hn in range(tl.cdiv(hn, block_size_hn)):
        # update block masks
        block_mask_hn_col = offsets_hn[:, None] < hn
        block_mask_hn_row = offsets_hn[None, :] < hn
        
        # load dr, lB
        dr = tl.load(dr_block_ptrs, mask=block_mask_hn_row, other=0.).to(compute_olB_dtype)
        lB = tl.load(lB_block_ptrs, mask=block_mask_hn_row, other=0.).to(compute_olB_dtype)
        
        # compute olB = lB @ dr.T
        olB += tl.dot(lB, dr.T).to(compute_olB_dtype) # shape = (rn, n)
        
        # update block pointers and offsets
        offsets_hn += block_size_hn
        dr_block_ptrs += block_size_hn
        lB_block_ptrs += block_size_hn
    
    ##################            mask olB            ##################
    
    # load lB_mask1, lB_mask2
    lB_mask1 = tl.load(lB_mask1_ptrs).to(compute_olB_dtype) # shape = (rn, n)
    lB_mask2 = tl.load(lB_mask2_ptrs, mask=block_mask_r_col, other=0.).to(compute_olB_dtype) # shape = (block_size_r, rn)
    # mask olB = lB_mask2 @ (olB * lB_mask1)
    olB = tl.dot(lB_mask2, olB * lB_mask1) # shape = (block_size_r, n)
    
    ##################      compute olA = lA @ olB     ##################
    compute_olA_dtype = tl.float16

    # load lA
    lA = tl.load(lA_block_ptrs, mask=block_mask_h_col & block_mask_r_row, other=0.).to(compute_olA_dtype) # shape = (block_size_h, block_size_r)
    # compute olA = lA @ olB
    olA = tl.sum(
        tl.dot(lA, olB.to(compute_olA_dtype)), # shape = (block_size_h, n)
        axis=1
    ) # shape = (block_size_h,)
    
    ##################           store output          ##################
    # store olA with shape = (block_size_h,)
    # TODO check the consistence between the shape of dx_block_ptrs and olA, check the store mask
    tl.store(dx_block_ptrs, olA.to(dx_block_ptrs.dtype.element_ty), mask=block_mask_h)


@triton.jit
def blora_bp_kernel(
        dr_ptr, dr_stride_bsk, dr_stride_m, dr_stride_hout,
        dx_ptr, dx_stride_bsk, dx_stride_m, dx_stride_h,
        blA_ptr, blA_stride_bsk, blA_stride_h, blA_stride_r,
        blB_ptr, blB_stride_bsk, blB_stride_r, blB_stride_hout,
        h: tl.constexpr, hout: tl.constexpr, m: tl.constexpr, r: tl.constexpr,
        block_size_h: tl.constexpr, block_size_hout: tl.constexpr,
        block_size_m: tl.constexpr, block_size_r: tl.constexpr,
    ):
    ##################      get the pointers for this bsk block      ##################
    # get the bsk block index and h block index
    block_idx_bsk = tl.program_id(0)
    block_idx_h = tl.program_id(1)
    
    # get block offsets
    offsets_h = block_idx_h * block_size_h + tl.arange(0, block_size_h)
    offsets_hout = tl.arange(0, block_size_hout)
    offsets_m = tl.arange(0, block_size_m)
    offsets_r = tl.arange(0, block_size_r)
    
    # get block masks
    block_mask_m_col = offsets_m[:, None] < m
    block_mask_r_row = offsets_r[None, :] < r
    block_mask_r_col = offsets_r[:, None] < r
    block_mask_h_row = offsets_h[None, :] < h
    block_mask_h_col = offsets_h[:, None] < h
    
    # get the variable pointers
    dx_ptrs = dx_ptr + block_idx_bsk * dx_stride_bsk + \
                (offsets_m[:, None] * dx_stride_m + offsets_h[None, :] * dx_stride_h) # shape = (block_size_m, block_size_h)
    dr_ptrs = dr_ptr + block_idx_bsk * dr_stride_bsk + \
                (offsets_m[:, None] * dr_stride_m + offsets_hout[None, :] * dr_stride_hout) # shape = (block_size_m, block_size_hout)
    blA_ptrs = blA_ptr + block_idx_bsk * blA_stride_bsk + \
                (offsets_h[:, None] * blA_stride_h + offsets_r[None, :] * blA_stride_r) # shape = (block_size_h, block_size_r)
    blB_ptrs = blB_ptr + block_idx_bsk * blB_stride_bsk + \
                (offsets_r[:, None] * blB_stride_r + offsets_hout[None, :] * blB_stride_hout) # shape = (block_size_r, block_size_hout)
    
    ##################      compute olB = blB @ dr.T      ##################
    olB = tl.zeros((block_size_r, block_size_m), dtype=tl.float32) # shape = (block_size_r, block_size_m)
    for block_idx_hout in range(0, tl.cdiv(hout, block_size_hout)):
        # get the offsets and masks for this block along inner dim hout
        block_offs_hout = block_idx_hout * block_size_hout + offsets_hout
        block_mask_hout_row = block_offs_hout[None, :] < hout
        block_mask_hout_col = block_offs_hout[:, None] < hout
        
        # load current block of x and blA along inner dimh from HBM to SRAM
        # with mask to avoid oob, where oob values are set to 0.
        dr = tl.load(dr_ptrs, mask=block_mask_m_col & block_mask_hout_row, other=0.) # shape = (block_size_m, block_size_hout)
        blB = tl.load(blB_ptrs, mask=block_mask_r_col & block_mask_hout_row, other=0.) # shape = (block_size_r, block_size_hout)
        
        # compute blB @ dr for this block along inner dim hout and add to `accumulator`
        olB += tl.dot(blB, dr.T) # shape = (block_size_r, block_size_m)
        
        # move dr,blB pointers to next block along inner dim hout
        dr_ptrs += block_size_hout * dr_stride_hout
        blB_ptrs += block_size_hout * blB_stride_hout
    
    ##################      compute olA = olA @ blB     ##################
    # load blA for this block along dim hout from HBM to SRAM
    blA = tl.load(blA_ptrs, mask=block_mask_h_col & block_mask_r_row, other=0.).to(tl.float32) # shape = (block_size_h, block_size_r)
    olA = tl.dot(blA, olB) # shape = (block_size_h, block_size_m)
    
    ##################      store output      ##################
    # store olA for this block along dim h from SRAM to HBM
    tl.store(dx_ptrs, olA.T, mask=block_mask_m_col & block_mask_h_row) # shape = (block_size_m, block_size_h)
    

@triton.jit
def moe_weigths_bp_kernel(
    dr_ptr, dr_stride_bsk, dr_stride_hout,
    dk_ptr, dk_stride_bsk, dk_stride_hout,
    dw_ptr, dw_stride_bsk, dw_stride_hout,
    hout: tl.constexpr, block_size_hout: tl.constexpr
):
    ##################      get the pointers for this bsk block      ##################
    # get the bsk block index and h block index
    block_idx_bsk = tl.program_id(0)
    block_idx_hout = tl.program_id(1)
    
    # get block offsets
    offsets_hout = block_idx_hout * block_size_hout + tl.arange(0, block_size_hout)
    
    # get block masks
    block_mask_hout = offsets_hout < hout
    
    # get the variable pointers
    dr_ptrs = dr_ptr + block_idx_bsk * dr_stride_bsk + offsets_hout * dr_stride_hout # shape = (block_size_hout, )
    dk_ptrs = dk_ptr + block_idx_bsk * dk_stride_bsk + offsets_hout * dk_stride_hout # shape = (block_size_hout, )
    dw_ptrs = dw_ptr + block_idx_bsk * dw_stride_bsk + offsets_hout * dw_stride_hout # shape = (block_size_hout, )
    
    ##################      compute dw = dr * dk      ##################
    # load current block of dr and dk from HBM to SRAM
    # with mask to avoid oob, where oob are set to 0.
    dr = tl.load(dr_ptrs, mask=block_mask_hout, other=0.) # shape = (block_size_hout, )
    dk = tl.load(dk_ptrs, mask=block_mask_hout, other=0.) # shape = (block_size_hout, )
    
    # compute dw = dr * dk for this block
    dw = dr * dk
    
    # store dw for this block along dim hout from SRAM to HBM
    tl.store(dw_ptrs, dw, mask=block_mask_hout)