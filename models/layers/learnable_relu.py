import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple

class LearnableReLU(nn.Module):
    """
    Slope-Controlled Learnable Activation Function.

    This module implements a monotonic, piecewise-smooth activation function that 
    preserves the Identity mapping for old tasks while allowing plasticity for new ones.
    
    Key Features:
    1. **Slope Control:** Guarantees f'(x) is always within [min_slope, max_slope].
    2. **Bi-Directional:** Can learn to amplify (slope > 1) or suppress (slope < 1) features.
    3. **Safety:** Hinge mechanism ensures f(x) = x for old task regions (Identity preservation).

    Math:
        f(x) = x + Sum( a_R * ReLU(x - c_R) ) - Sum( a_L * ReLU(c_L - x) )
        
        Where coefficients 'a' are derived from learned Target Slopes S_i:
        a_i = S_i - S_{i-1}
    """

    def __init__(
        self, 
        out_features: int, 
        k: int, 
        min_slope: float = 1e-3, 
        max_slope: float = 5.0
    ) -> None:
        """
        Args:
            out_features (int): Number of feature dimensions (e.g., 768).
            k (int): Maximum number of tasks (hinges).
            min_slope (float): Minimum allowed slope (must be > 0 for monotonicity).
            max_slope (float): Maximum allowed slope (controls plasticity limit).
        """
        super().__init__()

        self.k = k
        self.out_features = out_features
        self.min_slope = min_slope
        self.max_slope = max_slope
        self.no_curr_used_hinges = 0

        self.raw_slopes_r = nn.ParameterList(
            nn.Parameter(torch.zeros(1, out_features)) for _ in range(k - 1)
        )
        self.raw_slopes_l = nn.ParameterList(
            nn.Parameter(torch.zeros(1, out_features)) for _ in range(k - 1)
        )

        # Hinge positions (Anchors)
        self.register_buffer("c_r", torch.zeros(k - 1, 1, out_features))
        self.register_buffer("c_l", torch.zeros(k - 1, 1, out_features))

    def set_no_used_basis_functions(self, task_id: int) -> None:
        """
        Sets the number of active hinges based on the current task.
        task_id == number of active hinges.
        """
        self.no_curr_used_hinges = task_id

    def freeze_basis_function(self, idx: int) -> None:
        """
        Freeze the slope parameter for hinge idx (0-based).
        Used to lock plasticity for old tasks.
        """
        self.raw_slopes_r[idx].requires_grad_(False)
        self.raw_slopes_l[idx].requires_grad_(False)
        

    @torch.no_grad()
    def anchor_next_shift(
        self,
        z: torch.Tensor,
        task_id: int,
        percentile: float = 0.99,
    ) -> None:
        """
        Anchors the next hinge (task_id) based on the distribution of pre-activations 'z'.
        
        Args:
            z (Tensor): Batch of pre-activations from the current task.
            task_id (int): ID of the task being set up (1-based).
            percentile (float): Quantile to determine the hinge boundary (e.g., 0.99).
        """
        if z.dim() == 3:
            z = z.reshape(-1, z.size(-1))

        P_high = torch.quantile(z, percentile, dim=0, keepdim=True)
        P_low = torch.quantile(z, 1.0 - percentile, dim=0, keepdim=True)

        idx = task_id - 1

        if task_id == 1:
            self.c_r[idx] = P_high
            self.c_l[idx] = P_low
        else:
            self.c_r[idx] = torch.maximum(P_high, self.c_r[idx - 1] + 1e-4)
            self.c_l[idx] = torch.minimum(P_low,  self.c_l[idx - 1] - 1e-4)
            
        self.raw_slopes_r[idx].data.fill_(0.0)
        self.raw_slopes_l[idx].data.fill_(0.0)

    def _get_target_slopes(self, raw_params: nn.ParameterList) -> torch.Tensor:
        """
        Converts raw parameters into bounded Target Slopes S in [min, max].
        
        Mapping logic:
            p >= 0 -> Map to [1.0, max_slope] using Tanh
            p < 0  -> Map to [min_slope, 1.0] using Tanh
        """
        if not raw_params:
            return None
        
        # Stack parameters: [K-1, Features]
        P = torch.stack([p for p in raw_params], dim=0)
        
        delta_up = self.max_slope - 1.0
        delta_down = 1.0 - self.min_slope
        
        # Smoothly clamp P into the allowed slope ranges
        slopes = torch.where(
            P >= 0, 
            1.0 + torch.tanh(P) * delta_up,    # Amplification region
            1.0 + torch.tanh(P) * delta_down   # Suppression region
        )
        return slopes

    def get_coefficients(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculates the basis coefficients 'a' (change in slope) from Target Slopes.
        """
        # 1. Get Target Slopes for every region [K-1, 1, Feat]
        S_R = self._get_target_slopes(self.raw_slopes_r)
        S_L = self._get_target_slopes(self.raw_slopes_l)
        
        if S_R is None:
            return None, None
        
        # 2. Add the implicit "Region 0" slope (Identity = 1.0)
        base_slope = torch.ones(1, 1, self.out_features, device=S_R.device)
        
        # Prepend base slope -> [1.0, S_1, S_2, ...]
        # Concatenate along dim 0 (the 'hinge' dimension)
        full_S_R = torch.cat([base_slope, S_R], dim=0)
        full_S_L = torch.cat([base_slope, S_L], dim=0)
       
        # 3. Calculate Coefficients (Discrete Derivative of Slope)
        a_r = full_S_R[1:] - full_S_R[:-1]
        a_l = full_S_L[1:] - full_S_L[:-1]
        
        return a_r, a_l
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass applying the learned activation.
        """
        out = x

        if self.no_curr_used_hinges == 0:
            return out

        # 1. Get coefficients [Active, 1, Features]
        a_r_all, a_l_all = self.get_coefficients()
        
        a_r = a_r_all[:self.no_curr_used_hinges]
        a_l = a_l_all[:self.no_curr_used_hinges]
        c_r = self.c_r[:self.no_curr_used_hinges]
        c_l = self.c_l[:self.no_curr_used_hinges]

        # 2. Prepare Input for Broadcasting
        # x: [Batch, Features] -> [1, Batch, Features]
        x_u = x.unsqueeze(0)
        
        # 3. Handle 3D Inputs (e.g., [Batch, Seq, Features])
        if x.dim() == 3:
             a_r = a_r.unsqueeze(1); a_l = a_l.unsqueeze(1)
             c_r = c_r.unsqueeze(1); c_l = c_l.unsqueeze(1)
        
        # 4. Basis Functions (ReLU)        
        term_r = a_r * F.relu(x_u - c_r)
        term_l = a_l * F.relu(c_l - x_u)

        # 5. Sum over active hinges (dim 0)
        # Result shape matches x: [Batch, Features] (or [Batch, Seq, Features])
        out = out + term_r.sum(dim=0) - term_l.sum(dim=0)
        
        return out