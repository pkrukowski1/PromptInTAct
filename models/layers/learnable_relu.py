import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class LearnableReLU(nn.Module):
    """
    Adaptive Piecewise-Linear Activation (ReLU Base).

    Designed for Continual Learning in ViT backbones, this module extends a 
    standard ReLU with learnable "hinges" (breakpoints) that adaptively 
    modify the slope in specific regions of the feature space.
    
    This formulation guarantees strict monotonicity and non-negative slopes, 
    ensuring numerical stability for methods relying on Interval Bound 
    Propagation (IBP) or invertible flows.

    Key Mechanics:
    1. **Base Function:** f(x) = ReLU(x).
    2. **Plasticity:** Adds weighted hinge functions to the base to adjust 
       gradients for specific tasks.
    3. **Safety Constraint:** The total slope is strictly bounded in [0.01, 1.0]
       globally. This effectively converts the base ReLU into a Leaky ReLU 
       with learnable leakiness and saturation.
    
    Attributes:
        k (int): Maximum number of tasks/hinges.
        out_features (int): Dimensionality of the input features.
    """

    def __init__(self, out_features: int, k: int) -> None:
        """
        Initialize the Learnable Activation.

        Args:
            out_features (int): Number of feature dimensions (e.g., 768 for ViT-B).
            k (int): Maximum number of hinges/tasks to support.
        """
        super().__init__()

        self.k = k
        self.out_features = out_features
        self.num_active_hinges = 0

        # --- Learnable Parameters ---
        # RIGHT Side (Positive x): Initialize to -5.0
        # Sigmoid(-5.0) ~ 0.006 -> Target Slope = 1.0 - 0.99*0.006 ≈ 0.994.
        # Behavior: Standard Identity slope for positive values.
        self.raw_decay_r = nn.ParameterList(
            nn.Parameter(torch.ones(1, out_features) * -5.0) for _ in range(k - 1)
        )
        
        # LEFT Side (Negative x): Initialize to +5.0
        # Sigmoid(+5.0) ~ 0.993 -> Target Slope = 1.0 - 0.99*0.993 ≈ 0.017.
        # Behavior: Starts as Leaky ReLU (slope ~0.01) instead of dead zero.
        # This prevents "dead neurons" while respecting the 0.01 floor.
        self.raw_decay_l = nn.ParameterList(
            nn.Parameter(torch.ones(1, out_features) * 5.0) for _ in range(k - 1)
        )

        # Shape: [K-1, 1, D]. Batch/Token independent.
        self.register_buffer("c_r", torch.zeros(k - 1, 1, out_features))
        self.register_buffer("c_l", torch.zeros(k - 1, 1, out_features))

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
        Calculates and locks anchor positions (c_r, c_l) for the next task.
        
        Computes channel-wise statistics across all images (Batch) and 
        all patches (Tokens) to find global feature bounds.
        """
        # Collapse Batch and Token dimensions -> [N, Feat]
        # This treats every pixel in every image as a data point for statistics.
        if z.dim() == 3:
            z = z.reshape(-1, z.size(-1))

        # Calculate Channel-wise Percentiles [1, Feat]
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
        Calculates slope adjustment coefficients 'a'.
        
        Logic:
           a_r[i] = Slope_R[i+1] - Slope_R[i]
           a_l[i] = Slope_L[i+1] - Slope_L[i]
        """
        if self.num_active_hinges == 0:
            return None, None

        # 1. Stack parameters
        raw_r = torch.stack([self.raw_decay_r[i] for i in range(self.num_active_hinges)], dim=0)
        raw_l = torch.stack([self.raw_decay_l[i] for i in range(self.num_active_hinges)], dim=0)
        
        # 2. Cumulative Sum
        cum_r = torch.cumsum(raw_r, dim=0)
        cum_l = torch.cumsum(raw_l, dim=0)
        
        # 3. Target Slopes [0.01, 1.0]
        target_slope_r = 1.0 - 0.99 * torch.sigmoid(cum_r)
        target_slope_l = 1.0 - 0.99 * torch.sigmoid(cum_l)
        
        # 4. Define Base Slopes (ReLU Asymptotes)
        # ReLU Right Asymptote -> 1.0
        base_slope_r = torch.ones(1, 1, self.out_features, device=raw_r.device)
        
        # ReLU Left Asymptote -> 0.0
        # This allows calculating positive a_l to raise slope from 0.0 to 0.01
        base_slope_l = torch.zeros(1, 1, self.out_features, device=raw_l.device)
        
        # 5. Compute Deltas
        full_slope_r = torch.cat([base_slope_r, target_slope_r], dim=0)
        full_slope_l = torch.cat([base_slope_l, target_slope_l], dim=0)
        
        a_r = full_slope_r[1:] - full_slope_r[:-1]
        a_l = full_slope_l[1:] - full_slope_l[:-1]

        return a_r, a_l
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        f(x) = ReLU(x) + Sum(a_r * ReLU(x - c_r)) - Sum(a_l * ReLU(c_l - x))
        """
        # Base function: ReLU
        base = F.relu(x)

        if self.num_active_hinges == 0:
            return base

        # 1. Get Coefficients [K, 1, D]
        a_r, a_l = self.get_coefficients()
        
        # 2. Get Anchors [K, 1, D]
        c_r = self.c_r[:self.num_active_hinges]
        c_l = self.c_l[:self.num_active_hinges]

        # 3. Expand dimensions for ViT Broadcasting
        # Input: [Batch, Tokens, D] -> Target: [K, Batch, Tokens, D]
        x_u = x.unsqueeze(0)
        
        while a_r.dim() < x_u.dim():
            a_r = a_r.unsqueeze(1)
            a_l = a_l.unsqueeze(1)
            c_r = c_r.unsqueeze(1)
            c_l = c_l.unsqueeze(1)
        
        # 4. Apply Hinges
        # term_r adds slope for x > c_r
        term_r = a_r * F.relu(x_u - c_r)
        
        # term_l modifies slope for x < c_l
        # Note: We subtract term_l in the sum.
        # Since ReLU(c_l - x) has slope -1 w.r.t x, the subtraction adds +1 * a_l.
        term_l = a_l * F.relu(c_l - x_u)

        out = base + term_r.sum(dim=0) - term_l.sum(dim=0)
        
        return out