
import torch
import torch.nn.functional as F


@torch.compile
def blora_func_topk(x, blA, blB, lora_dropout, bscalings, moe_weights, bs, k):
    return (torch.bmm(torch.bmm(lora_dropout(x), blA), blB).squeeze(1) * bscalings * moe_weights).view(bs, k, -1).sum(dim=1)

@torch.compile
def blora_func_top1(x, blA, blB, lora_dropout, bscalings):
    return torch.bmm(torch.bmm(lora_dropout(x), blA), blB).squeeze(1) * bscalings

@torch.compile
def blora_func(x, blA, blB, lora_dropout):
    return torch.bmm(torch.bmm(lora_dropout(x), blA), blB)

# version 2: fuse torch.bmm(torch.bmm(x, blA), blB) and more operations with torch.compile
# however, we still use selected loras to index (bs*k) loras and do torch.bmm for each token with each topk
# FIXME: torch._dynamo hit config.cache_size_limit (8) and recompiled, so no speedup for version 2 compared to version 1
def _moelinear_fwd_inner_bmm_torch_v2(
        x: torch.Tensor, result: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor,  
    ) -> torch.Tensor:
    """the inner bmm process in MoELinear.forward implemented by torch
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
    # blora_results = blora_func(x, blora_A, blora_B, lora_dropout).squeeze(1) * bscalings
    
    # ## apply moe weighted sum for topk loras
    # if k > 1:
    #     blora_results *= moe_weights
    #     blora_results = blora_results.view(bs, k, -1).sum(dim=1) # shape from (bs*k, hout) to (bs, hout)
    
    if k > 1: blora_results = blora_func_topk(x, blora_A, blora_B, lora_dropout, bscalings, moe_weights, bs, k)
    else: blora_results = blora_func_top1(x, blora_A, blora_B, lora_dropout, bscalings)

    ## add the blora result to the base result
    result.add_(blora_results.to(x.dtype))
    
    return result
    

# version 1: use selected loras to index (bs*k) loras and do torch.bmm for each token with each topk
# which cost much more memory since there's only l different loras in the (bs*k) loras at most
def _moelinear_fwd_inner_bmm_torch_v1(
        x: torch.Tensor, result: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor,  
    ) -> torch.Tensor:
    """the inner bmm process in MoELinear.forward implemented by torch
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
    blora_results = torch.bmm( # shape: (bs*k, 1, h) => (bs*k, 1, r) => (bs*k, 1, hout) => (bs*k, hout)
        torch.bmm(lora_dropout(x), blora_A), blora_B
    ).squeeze(1) * bscalings
    
    ## apply moe weighted sum for topk loras
    if k > 1:
        blora_results *= moe_weights
        blora_results = blora_results.view(bs, k, -1).sum(dim=1) # shape from (bs*k, hout) to (bs, hout)

    ## add the blora result to the base result
    result.add_(blora_results.to(x.dtype))
    
    return result
    