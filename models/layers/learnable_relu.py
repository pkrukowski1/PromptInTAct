import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableReLU(nn.Module):
    """
    Normalized Learnable ReLU.

    Features:
    - Preserves variable names (w_r, w_l, c_r, c_l).
    - Uses Normalized Damping to prevent explosion with many tasks.
    - Uses Softplus on weights to ensure stability (monotonicity).
    
    Equation:
       Damping = Sum(Softplus(w) * Violation) / (1 + Sum(Softplus(w)))
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
        Sets the number of active hinges based on the current task.
        
        Args:
            task_id (int): Current task index (0-indexed). 
                           Task 0 uses pure ReLU.
                           Task 1 activates the first hinge set.
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

        # 1. Gather Parameters
        c_r_curr = self.c_r[:self.num_active_hinges]
        c_l_curr = self.c_l[:self.num_active_hinges]

        # Use Softplus to enforce positivity (Required for normalization stability)
        w_r_stack = torch.stack([self.w_r[i] for i in range(self.num_active_hinges)])
        w_l_stack = torch.stack([self.w_l[i] for i in range(self.num_active_hinges)])
        
        stiffness_r = F.softplus(w_r_stack)
        stiffness_l = F.softplus(w_l_stack)

        # 2. Broadcast
        x_u = x.unsqueeze(0)
        while stiffness_r.dim() < x_u.dim():
            stiffness_r = stiffness_r.unsqueeze(1)
            stiffness_l = stiffness_l.unsqueeze(1)
            c_r_curr = c_r_curr.unsqueeze(1)
            c_l_curr = c_l_curr.unsqueeze(1)

        # 3. Calculate Normalized Penalties
        
        # Right Side (Pull Down if x > c_r)
        excess_r = F.relu(x_u - c_r_curr)
        weighted_excess_r = stiffness_r * excess_r
        
        sum_penalty_r = weighted_excess_r.sum(dim=0)
        sum_stiffness_r = stiffness_r.sum(dim=0)
        
        # Note: Minus sign handled in the return statement
        damping_r = sum_penalty_r / (sum_stiffness_r + 1.0)

        # Left Side (Pull Up if x < c_l)
        excess_l = F.relu(c_l_curr - x_u)
        weighted_excess_l = stiffness_l * excess_l
        
        sum_penalty_l = weighted_excess_l.sum(dim=0)
        sum_stiffness_l = stiffness_l.sum(dim=0)
        
        damping_l = sum_penalty_l / (sum_stiffness_l + 1.0)

        # 4. Apply: Base - Down + Up
        return base - damping_r + damping_l