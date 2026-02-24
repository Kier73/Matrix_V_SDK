import torch
import torch.nn as nn
import math
from matrix_v_sdk.vl.substrate.matrix import MatrixOmega
from .utils import to_list, from_list


class MatrixVFunction(torch.autograd.Function):
    """
    Custom Autograd Function for Matrix-V acceleration.
    
    THEORY:
    Maps PyTorch tensor operations into the Virtual Layer (VL) manifolds. 
    The forward pass utilizes the optimal engine (Spectral, MMP, etc.).
    
    The backward pass supports two modes:
    - Standard: torch.mm (default, numerically stable)
    - RNS Exact: MMP_Engine for error-free gradient computation
      (enabled via exact_backward=True on MatrixVLinear)
    """
    @staticmethod
    def forward(ctx, input, weight, bias=None, omega=None, exact_backward=False):
        if omega is None:
            omega = MatrixOmega()
            
        ctx.save_for_backward(input, weight, bias)
        ctx.omega = omega
        ctx.exact_backward = exact_backward
        
        # Manifold Sharding: Handle Batch Dimension (Support 3D tensors)
        is_batched = input.dim() == 3
        if is_batched:
            batch_size, seq_len, in_feat = input.shape
            input_flat = input.view(-1, in_feat)
        else:
            input_flat = input

        # Convert to Virtual Layer (VL) format
        A = to_list(input_flat)
        B = to_list(weight.t()) 
        
        # Execute Virtualized Product
        res_list = omega.compute_product(A, B)
        
        output_flat = from_list(res_list, target_type='torch').to(input.device)
        
        if is_batched:
            output = output_flat.view(batch_size, seq_len, -1)
        else:
            output = output_flat

        if bias is not None:
            output += bias
            
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backprop through the VL-weighted graph.
        
        If exact_backward is set, uses MMP_Engine (RNS arithmetic)
        for dL/dW = grad^T @ input and dL/dX = grad @ weight.
        This preserves exact integer arithmetic through CRT,
        eliminating floating-point accumulation error in gradients.
        """
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.exact_backward:
            # RNS exact gradients via MMP_Engine
            omega = ctx.omega

            if ctx.needs_input_grad[0]:
                # dL/dX = grad_output @ weight
                g_list = to_list(grad_output)
                w_list = to_list(weight)
                gi_list = omega.mmp.multiply(g_list, w_list)
                grad_input = from_list(gi_list, target_type='torch').to(input.device)

            if ctx.needs_input_grad[1]:
                # dL/dW = grad_output^T @ input
                gt_list = to_list(grad_output.t())
                i_list = to_list(input)
                gw_list = omega.mmp.multiply(gt_list, i_list)
                grad_weight = from_list(gw_list, target_type='torch').to(input.device)
        else:
            # Standard torch gradients
            if ctx.needs_input_grad[0]:
                grad_input = grad_output.mm(weight)
            if ctx.needs_input_grad[1]:
                grad_weight = grad_output.t().mm(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias, None, None


class MatrixVLinear(nn.Module):
    """
    Virtualized Linear Layer.
    
    A drop-in replacement for nn.Linear that leverages the Matrix-V 
    acceleration stack. Ideal for trillion-scale layers or 
    high-rank spectral manifolds.
    
    Args:
        in_features: Input dimension
        out_features: Output dimension
        bias: Whether to include bias (default True)
        omega: MatrixOmega instance (shared across layers for cache)
        exact_backward: Use RNS exact arithmetic for gradients (default False).
            When True, gradient computation uses MMP_Engine with CRT reconstruction,
            eliminating floating-point error accumulation. Useful for:
            - Differential privacy (prevents FP rounding-based leakage)
            - Verifiable computation (exact gradients can be verified)
            - High-precision fine-tuning
    """
    def __init__(self, in_features, out_features, bias=True, omega=None, 
                 exact_backward=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.exact_backward = exact_backward
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.omega = omega if omega else MatrixOmega()

    def reset_parameters(self):
        """Standard Kaiming initialization for VL manifolds."""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        return MatrixVFunction.apply(
            input, self.weight, self.bias, self.omega, self.exact_backward
        )

