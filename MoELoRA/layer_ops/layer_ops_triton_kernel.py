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
def blora_fwd_kernel_with_loraA_mask(
        x_ptr, x_stride_bs, x_stride_m,
        o_ptr, o_stride_bsk,
        lA_ptr, lA_stride_l, lA_stride_hm,
        lB_ptr, lB_stride_l, lB_stride_r,
        lA_mask1_ptr, lA_mask1_stride_m,
        lA_mask2_ptr, lA_mask2_stride_rm,
        sel_ptr, 
        k: tl.constexpr, m: tl.constexpr, r: tl.constexpr, rm: tl.constexpr, 
        hm: tl.constexpr, hout: tl.constexpr,
        block_size_hout: tl.constexpr, block_size_hm: tl.constexpr, block_size_r: tl.constexpr,
    ):
    
    ##################       get the block pointers        ##################
    # get the bsk block index and hout block index
    block_idx_bsk = tl.program_id(0)
    block_idx_bs = block_idx_bsk // k
    block_idx_hout = tl.program_id(1)
    
    # get block offsets
    offsets_hout = block_idx_hout * block_size_hout + tl.arange(0, block_size_hout)
    offsets_hm = tl.arange(0, block_size_hm)
    offsets_r = tl.arange(0, block_size_r)
    offsets_m = tl.arange(0, m)
    offsets_rm = tl.arange(0, rm)
    
    # get block masks
    block_mask_hout = offsets_hout < hout
    block_mask_hout_row = offsets_hout[None, :] < hout
    block_mask_r_col = offsets_r[:, None] < r
    block_mask_r_row = offsets_r[None, :] < r
    
    # get the selected lora idx
    sel_ptr += block_idx_bsk
    sel_idx = tl.load(sel_ptr)
    
    # get the block pointers for input variables: x, lA, lB, lA_mask1, lA_mask2
    x_block_ptrs = x_ptr + block_idx_bs * x_stride_bs + \
                (offsets_m[:, None] * x_stride_m + offsets_hm[None, :]) # shape = (m, block_size_hm)
    lA_block_ptrs = lA_ptr + sel_idx * lA_stride_l + \
                (offsets_hm[:, None] * lA_stride_hm + offsets_rm[None, :]) # shape = (block_size_hm, rm)
    lB_block_ptrs = lB_ptr + sel_idx * lB_stride_l + \
                (offsets_r[:, None] * lB_stride_r + offsets_hout[None, :]) # shape = (block_size_r, block_size_hout)
    lA_mask1_ptrs = lA_mask1_ptr + (offsets_m[:, None] * lA_mask1_stride_m + offsets_rm[None, :]) # shape = (m, rm)
    lA_mask2_ptrs = lA_mask2_ptr + (offsets_rm[:, None] * lA_mask2_stride_rm + offsets_r[None, :]) # shape = (rm, block_size_r)
                
    # get the block pointers for output variables
    o_block_ptrs = o_ptr + block_idx_bsk * o_stride_bsk + offsets_hout # shape = (block_size_hout, )
    
    ##################      compute olA = x @ lA      ##################
    # compute_olA_dtype = tl.float32
    compute_olA_dtype = tl.float16
    # compute_olA_dtype = o_block_ptrs.dtype.element_ty
    
    # init olA buffer
    olA = tl.zeros((m, rm), dtype=compute_olA_dtype) # shape = (m, rm)
    # loop over hm dim by blocks to accumulate olA
    for block_idx_hm in range(tl.cdiv(hm, block_size_hm)):
        # update block masks
        block_mask_hm_col = offsets_hm[:, None] < hm
        block_mask_hm_row = offsets_hm[None, :] < hm
        
        # load x, lA
        x = tl.load(x_block_ptrs, mask=block_mask_hm_row, other=0.).to(compute_olA_dtype) # shape = (m, block_size_hm)
        lA = tl.load(lA_block_ptrs, mask=block_mask_hm_col, other=0.).to(compute_olA_dtype) # shape = (block_size_hm, rm)
        
        # compute olA = x @ lA
        olA += tl.dot(x, lA).to(compute_olA_dtype) # shape = (m, rm)
        
        # update block pointers and offsets
        offsets_hm += block_size_hm
        x_block_ptrs += block_size_hm
        lA_block_ptrs += block_size_hm * lA_stride_hm
    
    
    ##################            mask olA            ##################
    
    # load lA_mask1, lA_mask2
    lA_mask1 = tl.load(lA_mask1_ptrs).to(compute_olA_dtype) # shape = (m, rm)
    lA_mask2 = tl.load(lA_mask2_ptrs, mask=block_mask_r_row, other=0.).to(compute_olA_dtype) # shape = (rm, block_size_r)
    # mask olA = olA * lA_mask1 @ lA_mask2
    olA = tl.dot(olA * lA_mask1, lA_mask2) # shape = (m, block_size_r)
    
    ##################      compute olB = olA @ lB     ##################
    # compute_olB_dtype = tl.float32
    compute_olB_dtype = tl.float16
    # compute_olB_dtype = o_block_ptrs.dtype.element_ty
    
    # load lB
    lB = tl.load(lB_block_ptrs, mask=block_mask_r_col & block_mask_hout_row, other=0.).to(compute_olB_dtype) # shape = (block_size_r, block_size_hout)
    # compute olB = olA @ lB
    olB = tl.sum( # shape = (block_size_hout, )
        tl.dot(olA.to(compute_olB_dtype), lB), # shape = (m, block_size_hout)
        axis=0,
    )
    
    ##################           store output          ##################
    # store olB with shape = (block_size_hout,)
    tl.store(o_block_ptrs, olB.to(o_block_ptrs.dtype.element_ty), mask=block_mask_hout)


@triton.jit
def blora_fwd_kernel(
        x_ptr, x_stride_bsk, x_stride_m, x_stride_h,
        o_ptr, o_stride_bsk, o_stride_m, o_stride_hout,
        blA_ptr, blA_stride_bsk, blA_stride_h, blA_stride_r,
        blB_ptr, blB_stride_bsk, blB_stride_r, blB_stride_hout,
        h: tl.constexpr, hout: tl.constexpr, m: tl.constexpr, r: tl.constexpr,
        block_size_h: tl.constexpr, block_size_hout: tl.constexpr,
        block_size_m: tl.constexpr, block_size_r: tl.constexpr,
    ):
    
    ##################      get the pointers for this bsk block      ##################
    # get the bsk block index and hout block index
    block_idx_bsk = tl.program_id(0)
    block_idx_hout = tl.program_id(1)
    
    # get block offsets
    offsets_h = tl.arange(0, block_size_h)
    offsets_hout = block_idx_hout * block_size_hout + tl.arange(0, block_size_hout)
    offsets_m = tl.arange(0, block_size_m)
    offsets_r = tl.arange(0, block_size_r)
    
    # get block masks
    block_mask_m_col = offsets_m[:, None] < m
    block_mask_r_row = offsets_r[None, :] < r
    block_mask_r_col = offsets_r[:, None] < r
    block_mask_hout_row = offsets_hout[None, :] < hout
    
    # get the variable pointers
    x_ptrs = x_ptr + block_idx_bsk * x_stride_bsk + \
                (offsets_m[:, None] * x_stride_m + offsets_h[None, :] * x_stride_h) # shape = (block_size_m, block_size_h)
    o_ptrs = o_ptr + block_idx_bsk * o_stride_bsk + \
                (offsets_m[:, None] * o_stride_m + offsets_hout[None, :] * o_stride_hout) # shape = (block_size_m, block_size_hout)
    blA_ptrs = blA_ptr + block_idx_bsk * blA_stride_bsk + \
                (offsets_h[:, None] * blA_stride_h + offsets_r[None, :] * blA_stride_r) # shape = (block_size_h, block_size_r)
    blB_ptrs = blB_ptr + block_idx_bsk * blB_stride_bsk + \
                (offsets_r[:, None] * blB_stride_r + offsets_hout[None, :] * blB_stride_hout) # shape = (block_size_r, block_size_hout)
    
    ##################      compute olA = x @ blA      ##################
    olA = tl.zeros((block_size_m, block_size_r), dtype=tl.float32) # shape = (block_size_m, block_size_r), use tl.float32 to get higher precision
    for block_idx_h in range(0, tl.cdiv(h, block_size_h)): # accumulate olA along inner dimh block by block
        # get the offsets and masks for this block along inner dimh
        block_offs_h = block_idx_h * block_size_h + offsets_h
        block_mask_h_row = block_offs_h[None, :] < h
        block_mask_h_col = block_offs_h[:, None] < h
        
        # load current block of x and blA along inner dimh from HBM to SRAM
        # with mask to avoid oob, where oob values are set to 0.
        x = tl.load(x_ptrs, mask=block_mask_m_col & block_mask_h_row, other=0.) # shape = (block_size_m, block_size_h)
        blA = tl.load(blA_ptrs, mask=block_mask_h_col & block_mask_r_row, other=0.) # shape = (block_size_h, block_size_r)
        
        # compute x @ blA for this block along inner dimh and add to `accumulator`
        olA += tl.dot(x, blA)
        
        # move x,blA pointers to next block along inner dimh
        x_ptrs += block_size_h * x_stride_h
        blA_ptrs += block_size_h * blA_stride_h
    
    ##################      compute olB = olA @ blB     ##################
    # load blB for this block along dim hout from HBM to SRAM
    blB = tl.load(blB_ptrs, mask=block_mask_r_col & block_mask_hout_row, other=0.).to(tl.float32) # shape = (block_size_r, block_size_hout), upcast to float32 to get higher precision
    olB = tl.dot(olA, blB) # shape = (block_size_m, block_size_hout)
    
    ##################      store output      ##################
    # store olB for this block along dim hout from SRAM to HBM
    tl.store(o_ptrs, olB, mask=block_mask_m_col & block_mask_hout_row) # shape = (block_size_m, block_size_hout)
