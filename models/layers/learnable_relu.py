import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableReLU(nn.Module):
    """
    Normalized Additive Learnable ReLU.

    A dynamic activation function that introduces learnable piecewise-linear 
    corrections (hinges) for Continual Learning.

    This module provides **Plasticity**:
    - It adds new basis functions (hinges) at the boundaries of previous tasks.
    - Weights (w) can be positive (slope up) or negative (slope down).
    - It enables the model to learn new high-magnitude features for new tasks 
      without destabilizing the scale of the output.

    Equation:
       Correction = Sum(w * ReLU(Violation)) / (1 + Sum(|w|))
       f(x) = Base(x) + Correction
    """

    def __init__(self, out_features: int, k: int, base_function: nn.Module = torch.nn.Identity()) -> None:
        """
        Initialize the Normalized Activation.

        Args:
            out_features (int): Feature dimensions.
            k (int): Maximum number of tasks.
            base_function (nn.Module): Base activation function (default: Identity).
        """
        super().__init__()

        self.k = k
        self.out_features = out_features
        self.num_active_hinges = 0
        self.base_function = base_function

        self.w_r = nn.ParameterList(
            nn.Parameter(torch.zeros(1, out_features)) for _ in range(k - 1)
        )
        self.w_l = nn.ParameterList(
            nn.Parameter(torch.zeros(1, out_features)) for _ in range(k - 1)
        )

        self.register_buffer("c_r", torch.ones(k - 1, 1, out_features) * 9999.0)
        self.register_buffer("c_l", torch.ones(k - 1, 1, out_features) * -9999.0)

    def set_no_used_basis_functions(self, task_id: int) -> None:
        """
        Activates the hinges for the specified task.
        
        Args:
            task_id (int): Current task index (0-indexed). 
                           Task 0: Pure Base Function.
                           Task 1: Base + 1 Hinge, etc.
        """
        self.num_active_hinges = task_id

    def freeze_basis_function(self, idx: int) -> None:
        """
        Freeze the parameters for hinge `idx` to preserve memory of old tasks.
        """
        self.w_r[idx].requires_grad_(False)
        self.w_l[idx].requires_grad_(False)
            
    @torch.no_grad()
    def anchor_next_shift(
        self,
        z: torch.Tensor,
        task_id: int,
        percentile: float = 0.99,
    ) -> None:
        """
        Calculates and locks anchor positions (c_r, c_l) for the next task.
        
        Computes channel-wise statistics across all images (Batch) and 
        all patches (Tokens) to find global feature bounds.
        """
        if z.dim() == 3:
            z = z.reshape(-1, z.size(-1))

        P_high = torch.quantile(z, percentile, dim=0, keepdim=True)
        P_low = torch.quantile(z, 1.0 - percentile, dim=0, keepdim=True)
        
        idx = task_id - 1
        
        if idx >= 0 and idx < len(self.c_r):
            if task_id == 1:
                self.c_r[idx] = P_high
                self.c_l[idx] = P_low
            else:
                self.c_r[idx] = torch.maximum(P_high, self.c_r[idx-1])
                self.c_l[idx] = torch.minimum(P_low,  self.c_l[idx-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_function(x)

        if self.num_active_hinges == 0:
            return base

        c_r_curr = self.c_r[:self.num_active_hinges]
        c_l_curr = self.c_l[:self.num_active_hinges]

        w_r_stack = torch.stack([self.w_r[i] for i in range(self.num_active_hinges)])
        w_l_stack = torch.stack([self.w_l[i] for i in range(self.num_active_hinges)])
        
        x_u = x.unsqueeze(0)
        while w_r_stack.dim() < x_u.dim():
            w_r_stack = w_r_stack.unsqueeze(1)
            w_l_stack = w_l_stack.unsqueeze(1)
            c_r_curr = c_r_curr.unsqueeze(1)
            c_l_curr = c_l_curr.unsqueeze(1)

        hinge_r = F.relu(x_u - c_r_curr)
        hinge_l = F.relu(c_l_curr - x_u)

        correction_r = w_r_stack * hinge_r
        correction_l = w_l_stack * hinge_l
        
        sum_correction = correction_r.sum(dim=0) + correction_l.sum(dim=0)
        
        sum_abs_w = w_r_stack.abs().sum(dim=0) + w_l_stack.abs().sum(dim=0)
        normalization = 1.0 + sum_abs_w
        
        return base + (sum_correction / normalization)