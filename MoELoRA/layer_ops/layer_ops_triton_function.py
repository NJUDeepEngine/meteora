import torch
from typing import Tuple
from .layer_ops_triton_backward import _moelinear_bp_inner_bmm_triton_weights

################################### v4 ####################################
from .layer_ops_triton import _moelinear_fwd_inner_bmm_triton_v4
from .layer_ops_triton_backward import _moelinear_bp_inner_bmm_triton_v4

class MoeLinear_Inner_Bmm_Triton_v4(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, result: torch.Tensor,
                lora_A_weights: torch.Tensor, lora_A_weights_split: torch.Tensor,
                lora_B_weights: torch.Tensor, lora_B_weights_split: torch.Tensor,
                scalings: torch.Tensor, lora_dropout: torch.nn.Module,
                moe_weights: torch.Tensor, selected_loras: torch.Tensor,
                lora_A_mask: Tuple[torch.Tensor, torch.Tensor],
                lora_B_mask: Tuple[torch.Tensor, torch.Tensor]
            ) -> torch.Tensor:
        
        kwargs = dict(
            x=x, result=result,
            lora_A_weights=lora_A_weights_split, lora_B_weights=lora_B_weights,
            scalings=scalings, lora_dropout=lora_dropout,
            moe_weights=moe_weights, selected_loras=selected_loras,
            lora_A_mask=lora_A_mask
        )
        _, result_moe = _moelinear_fwd_inner_bmm_triton_v4(**kwargs)
        
        x_shape_tensor = torch.IntTensor([x.shape[0], x.shape[1]])
        
        ctx.save_for_backward(x_shape_tensor, lora_A_weights, lora_B_weights_split,
                              scalings, lora_dropout, moe_weights, selected_loras,
                              lora_B_mask[0], lora_B_mask[1], result_moe)
        return result
    
    @staticmethod
    def backward(ctx, grad_outputs):
        x_shape_tensor, lora_A_weights, lora_B_weights,\
                scalings, lora_dropout, moe_weights, selected_loras,\
                              lora_B_mask1, lora_B_mask2, result_moe, = ctx.saved_tensors
        # dx = torch.zeros(x_shape, dtype=grad_outputs.type, device=grad_outputs.device)
        # dx = torch.zeros_like(x)
        dx = torch.zeros((x_shape_tensor[0].item(), x_shape_tensor[1].item()), dtype=grad_outputs.dtype, device=grad_outputs.device)
        dmoe = torch.zeros_like(moe_weights)
        
        kwargs = dict(
            dr=grad_outputs, dx=dx,
            lora_A_weights=lora_A_weights, lora_B_weights=lora_B_weights,
            scalings=scalings, lora_dropout=lora_dropout,
            moe_weights=moe_weights, selected_loras=selected_loras,
            lora_B_mask=(lora_B_mask1, lora_B_mask2)
        )
        
        _moelinear_bp_inner_bmm_triton_v4(**kwargs)
        _moelinear_bp_inner_bmm_triton_weights(grad_outputs, result_moe, dmoe, moe_weights)
        
        return dx, None, None, None, None, None, None, None, dmoe, None, None, None 
        

################################### v3 ####################################
from .layer_ops_triton import _moelinear_fwd_inner_bmm_triton_v3
from .layer_ops_triton_backward import _moelinear_bp_inner_bmm_triton_v3

class MoeLinear_Inner_Bmm_Triton_v3(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, result: torch.Tensor,
                lora_A_weights: torch.Tensor, lora_A_weights_split: torch.Tensor,
                lora_B_weights: torch.Tensor, lora_B_weights_split: torch.Tensor,
                scalings: torch.Tensor, lora_dropout: torch.nn.Module,
                moe_weights: torch.Tensor, selected_loras: torch.Tensor,
                lora_A_mask: torch.Tensor,
                lora_B_mask: torch.Tensor
            ) -> torch.Tensor:
        
        kwargs = dict(
            x=x, result=result,
            lora_A_weights=lora_A_weights_split, lora_B_weights=lora_B_weights,
            scalings=scalings, lora_dropout=lora_dropout,
            moe_weights=moe_weights, selected_loras=selected_loras,
            lora_A_mask=lora_A_mask
        )
        _, result_moe = _moelinear_fwd_inner_bmm_triton_v3(**kwargs)
        
        x_shape_tensor = torch.IntTensor([x.shape[0], x.shape[1]])
        
        ctx.save_for_backward(x_shape_tensor, lora_A_weights, lora_B_weights_split,
                              scalings, lora_dropout, moe_weights, selected_loras,
                              lora_B_mask, result_moe)
        return result
    
    @staticmethod
    def backward(ctx, grad_outputs):
        x_shape_tensor, lora_A_weights, lora_B_weights,\
                scalings, lora_dropout, moe_weights, selected_loras,\
                              lora_B_mask, result_moe, = ctx.saved_tensors
        # dx = torch.zeros(x_shape, dtype=grad_outputs.type, device=grad_outputs.device)
        # dx = torch.zeros_like(x)
        dx = torch.zeros((x_shape_tensor[0].item(), x_shape_tensor[1].item()), dtype=grad_outputs.dtype, device=grad_outputs.device)
        dmoe = torch.zeros_like(moe_weights)
        
        kwargs = dict(
            dr=grad_outputs, dx=dx,
            lora_A_weights=lora_A_weights, lora_B_weights=lora_B_weights,
            scalings=scalings, lora_dropout=lora_dropout,
            moe_weights=moe_weights, selected_loras=selected_loras,
            lora_B_mask=lora_B_mask
        )
        
        _moelinear_bp_inner_bmm_triton_v3(**kwargs)
        _moelinear_bp_inner_bmm_triton_weights(grad_outputs, result_moe, dmoe, moe_weights)
        
        return dx, None, None, None, None, None, None, None, dmoe, None, None, None 

################################### v2 ####################################
from .layer_ops_triton import _moelinear_fwd_inner_bmm_triton_v2
from .layer_ops_triton_backward import _moelinear_bp_inner_bmm_triton_v2

class MoeLinear_Inner_Bmm_Triton_v2(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, result: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor):
        
        kwargs = dict(
            x=x, result=result,
            lora_A_weights=lora_A_weights, lora_B_weights=lora_B_weights,
            scalings=scalings, lora_dropout=lora_dropout,
            moe_weights=moe_weights, selected_loras=selected_loras,
        )
        _, result_moe = _moelinear_fwd_inner_bmm_triton_v2(**kwargs)
        x_shape_tensor = torch.IntTensor([x.shape[0], x.shape[1]])
        ctx.save_for_backward(x_shape_tensor, lora_A_weights, lora_B_weights, scalings, 
                              moe_weights, selected_loras, result_moe)
        return result
    
    @staticmethod
    def backward(ctx, grad_outputs):
        x_shape_tensor, lora_A_weights, lora_B_weights, scalings, \
            moe_weights, selected_loras, blora_result = ctx.saved_tensors
        # dx = torch.zeros_like(x)
        dx = torch.zeros((x_shape_tensor[0].item(), x_shape_tensor[1].item()), dtype=grad_outputs.dtype, device=grad_outputs.device)
        dmoe = torch.zeros_like(moe_weights)
        
        kwargs = dict(
            dr=grad_outputs, dx=dx,
            lora_A_weights=lora_A_weights, lora_B_weights=lora_B_weights,
            scalings=scalings, lora_dropout=None,
            moe_weights=moe_weights, selected_loras=selected_loras,
        )
        
        _moelinear_bp_inner_bmm_triton_v2(**kwargs)
        _moelinear_bp_inner_bmm_triton_weights(grad_outputs, blora_result, dmoe, moe_weights)
        return dx, None, None, None, None, None, dmoe, None


from .layer_ops_triton import _moelinear_fwd_inner_bmm_triton_v1
from .layer_ops_triton_backward import _moelinear_bp_inner_bmm_triton_v1

################################### v1 ####################################
class MoeLinear_Inner_Bmm_Triton_v1(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x: torch.Tensor, result: torch.Tensor, 
        lora_A_weights: torch.Tensor, lora_B_weights: torch.Tensor, 
        scalings: torch.Tensor, lora_dropout: torch.nn.Module,
        moe_weights: torch.Tensor, selected_loras: torch.Tensor):
        
        kwargs = dict(
            x=x, result=result,
            lora_A_weights=lora_A_weights, lora_B_weights=lora_B_weights,
            scalings=scalings, lora_dropout=lora_dropout,
            moe_weights=moe_weights, selected_loras=selected_loras,
        )
        _, result_moe = _moelinear_fwd_inner_bmm_triton_v1(**kwargs)
        x_shape_tensor = torch.IntTensor([x.shape[0], x.shape[1]])
        ctx.save_for_backward(x_shape_tensor, lora_A_weights, lora_B_weights, scalings, 
                              moe_weights, selected_loras, result_moe)
        return result
    
    @staticmethod
    def backward(ctx, grad_outputs):
        x_shape_tensor, lora_A_weights, lora_B_weights, scalings, \
            moe_weights, selected_loras, blora_result = ctx.saved_tensors
        # dx = torch.zeros_like(x)
        dx = torch.zeros((x_shape_tensor[0].item(), x_shape_tensor[1].item()), dtype=grad_outputs.dtype, device=grad_outputs.device)
        dmoe = torch.zeros_like(moe_weights)
        
        kwargs = dict(
            dr=grad_outputs, dx=dx,
            lora_A_weights=lora_A_weights, lora_B_weights=lora_B_weights,
            scalings=scalings, lora_dropout=None,
            moe_weights=moe_weights, selected_loras=selected_loras,
        )
        
        _moelinear_bp_inner_bmm_triton_v1(**kwargs)
        _moelinear_bp_inner_bmm_triton_weights(grad_outputs, blora_result, dmoe, moe_weights)
        return dx, None, None, None, None, None, dmoe, None