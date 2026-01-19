import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class LearnableReLU(nn.Module):
    """
    Adaptive Sublinear ReLU for Continual Learning.

    This activation function is designed for high-difficulty benchmarks like ImageNet-R.
    It provides a strictly monotonic, piecewise-linear response that can adaptively 
    suppress noise (slope < 1) but never amplify features (slope <= 1), ensuring 
    stability for Interval Bound Propagation (InTAct).

    Key Mechanics:
    1. **Base Function:** f(x) = x.
    2. **Plasticity:** Adds splines to the base function to modify the slope for specific regions.
    3. **Safety constraint:** Slope is calculated via Sigmoid of cumulative parameters, 
       guaranteeing f'(x) is always in [0.01, 1.0].
    
    Attributes:
        k (int): Maximum number of hinges (tasks * hinges_per_task).
        out_features (int): Dimensionality of the input features.
    """

    def __init__(self, out_features: int, k: int) -> None:
        """
        Args:
            out_features (int): Number of feature dimensions (e.g., 768).
            k (int): Maximum number of hinges to support (e.g., 15 tasks * 1 hinge = 15).
        """
        super().__init__()

        self.k = k
        self.out_features = out_features
        self.no_curr_used_hinges = 0

        # --- Learnable Parameters ---
        # We initialize to -5.0. 
        # Sigmoid(-5.0) approx 0.006. 
        # Resulting Slope = 1.0 - 0.99 * 0.006 approx 0.994 (Identity).
        # This ensures the model starts neutral for every new task.
        self.raw_decay_r = nn.ParameterList(
            nn.Parameter(torch.ones(1, out_features) * -5.0) for _ in range(k - 1)
        )
        self.raw_decay_l = nn.ParameterList(
            nn.Parameter(torch.ones(1, out_features) * -5.0) for _ in range(k - 1)
        )

        # --- Fixed Anchors (Buffers) ---
        self.register_buffer("c_r", torch.zeros(k - 1, 1, out_features))
        self.register_buffer("c_l", torch.zeros(k - 1, 1, out_features))

    def set_no_used_basis_functions(self, task_id: int) -> None:
        """
        Sets the number of active hinges based on the current task.
        
        Args:
            task_id (int): 0-indexed task ID. 
                           Task 0 uses identity.
                           Task 1 uses 1 hinge (identity + Spline 1).
        """
        self.no_curr_used_hinges = task_id

    def freeze_basis_function(self, idx: int) -> None:
        """
        Freeze the parameters for hinge `idx` to preserve memory of old tasks.
        """
        if idx < len(self.raw_decay_r):
            self.raw_decay_r[idx].requires_grad_(False)
            self.raw_decay_l[idx].requires_grad_(False)
            
    @torch.no_grad()
    def anchor_next_shift(
        self,
        z: torch.Tensor,
        task_id: int,
        percentile: float = 0.99,
    ) -> None:
        """
        Calculates and locks the anchor positions (c_r, c_l) for the next task
        based on the feature distribution of the current batch.
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

    def get_coefficients(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Calculates the basis coefficients 'a' derived from the Adaptive Sigmoid logic.
        
        Logic:
           Slope[i] = 1.0 - 0.99 * Sigmoid( CumulativeSum(Params_0...Params_i) )
        
        Returns:
            a_r, a_l: Coefficients for the splines.
        """
        if self.no_curr_used_hinges == 0:
            return None, None

        # 1. Stack parameters for currently active hinges
        # [Active_Hinges, 1, Features]
        raw_r = torch.stack([self.raw_decay_r[i] for i in range(self.no_curr_used_hinges)], dim=0)
        raw_l = torch.stack([self.raw_decay_l[i] for i in range(self.no_curr_used_hinges)], dim=0)
        
        # 2. Compute Cumulative Sum ("Total Pressure")
        cum_r = torch.cumsum(raw_r, dim=0)
        cum_l = torch.cumsum(raw_l, dim=0)
        
        # 3. Apply Adaptive Sigmoid Bound
        # Guarantees slope is strictly in [0.01, 1.0]
        target_slope_r = 1.0 - 0.99 * torch.sigmoid(cum_r)
        target_slope_l = 1.0 - 0.99 * torch.sigmoid(cum_l)
        
        # 4. Compute Coefficients (Difference between consecutive slopes)
        # We prepend the Base Slope (1.0).
        base = torch.ones(1, 1, self.out_features, device=raw_r.device)
        
        full_slope_r = torch.cat([base, target_slope_r], dim=0)
        full_slope_l = torch.cat([base, target_slope_l], dim=0)
        
        # a[i] = Slope[i+1] - Slope[i]
        a_r = full_slope_r[1:] - full_slope_r[:-1]
        a_l = full_slope_l[1:] - full_slope_l[:-1]

        return a_r, a_l
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        f(x) = x + Sum(a_r * ReLU(x - c_r)) - Sum(a_l * ReLU(c_l - x))
        """
        base = x

        if self.no_curr_used_hinges == 0:
            return base

        # 1. Get Coefficients
        a_r, a_l = self.get_coefficients()
        
        # 2. Get Anchors (slice for active tasks)
        c_r = self.c_r[:self.no_curr_used_hinges]
        c_l = self.c_l[:self.no_curr_used_hinges]

        # 3. Reshape for Broadcasting
        x_u = x.unsqueeze(0)
        
        if x.dim() == 3: # Handle Sequence Data [Batch, Seq, Feat]
             a_r = a_r.unsqueeze(1); a_l = a_l.unsqueeze(1)
             c_r = c_r.unsqueeze(1); c_l = c_l.unsqueeze(1)
        
        # 4. Apply Splines to Base
        term_r = a_r * F.relu(x_u - c_r)
        term_l = a_l * F.relu(c_l - x_u)

        out = base + term_r.sum(dim=0) - term_l.sum(dim=0)
        
        return out