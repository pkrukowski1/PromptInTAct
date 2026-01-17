import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableReLU(nn.Module):
    """
    LearnableReLU: Monotone-by-construction activation via hinge expansion.

    Task semantics:
        • Task 0: identity (no hinges)
        • Each new task adds one hinge
        • Maximum number of tasks = k
        • Number of hinges = k - 1
    """

    def __init__(self, out_features: int, k: int) -> None:
        super().__init__()

        self.k = k
        self.out_features = out_features
        self.no_curr_used_hinges = 0

        # We learn the slopes for the Left and Right regions explicitly.
        # Initialized to give a slope of ~1.0 (Softplus(0.54) approx 1.0)
        # to start close to Identity.
        init_val = 0.5413 
        
        self.raw_slopes_r = nn.ParameterList(
            nn.Parameter(torch.ones(1, out_features) * init_val) for _ in range(k - 1)
        )
        self.raw_slopes_l = nn.ParameterList(
            nn.Parameter(torch.ones(1, out_features) * init_val) for _ in range(k - 1)
        )

        self.register_buffer("base_slope", torch.ones(1, out_features))

        self.register_buffer("c_r", torch.zeros(k - 1, 1, out_features))
        self.register_buffer("c_l", torch.zeros(k - 1, 1, out_features))

    def set_no_used_basis_functions(self, task_id: int) -> None:
        """
        task_id == number of active hinges
        """
        self.no_curr_used_hinges = task_id

    def freeze_basis_function(self, idx: int) -> None:
        """
        Freeze hinge idx (0-based).
        """
        self.raw_slopes_r[idx].requires_grad_(False)
        self.raw_slopes_l[idx].requires_grad_(False)
        

    @torch.no_grad()
    def anchor_next_shift(
        self,
        z: torch.Tensor,
        task_id: int,
        percentile: float = 0.95,
    ) -> None:
        """
        Anchors hinge (task_id - 1) using PREACTIVATIONS.
        """

        if z.dim() == 3:
            z = z.reshape(-1, z.size(-1))

        P_high = torch.quantile(z, percentile, dim=0, keepdim=True)
        P_low = torch.quantile(z, 1.0 - percentile, dim=0, keepdim=True)

        idx = task_id - 1  # hinge index

        if task_id == 1:
            self.c_r[idx] = P_high
            self.c_l[idx] = P_low
        else:
            self.c_r[idx] = torch.maximum(P_high, self.c_r[idx - 1])
            self.c_l[idx] = torch.minimum(P_low,  self.c_l[idx - 1])

    def get_coefficients(self):
        """
        Convert positive slopes into additive coefficients for ReLUs.
        
        Slope Sequence: ... s_L2, s_L1, (base=1), s_R1, s_R2 ...
        
        The coefficient 'a' for a hinge is the CHANGE in slope.
        a_R[i] = slope_R[i] - slope_R[i-1]
        """
        
        # 1. Enforce positivity of slopes
        s_r = [F.softplus(p) for p in self.raw_slopes_r]
        s_l = [F.softplus(p) for p in self.raw_slopes_l]
        
        if not s_r:
            return None, None

        S_R = torch.stack(s_r, dim=0)
        S_L = torch.stack(s_l, dim=0)

        # 2. Prepend the "previous" slope. 
        
        # Right side: slopes define segments moving away from center
        prev_S_R = torch.cat([self.base_slope.unsqueeze(0), S_R[:-1]], dim=0)
        a_r = S_R - prev_S_R
        
        # Left side: slopes define segments moving away from center (towards -inf)
        # Note: The "direction" of integration for the formula requires careful sign handling.
        # Formula: - a_L * ReLU(c_L - x)
        # As x decreases (moves left), the slope becomes (Base + Sum(a_L)).
        # We want the resulting slope to match S_L.
        prev_S_L = torch.cat([self.base_slope.unsqueeze(0), S_L[:-1]], dim=0)
        a_l = S_L - prev_S_L

        return a_r, a_l
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        f(x) = x 
             + Σ (s_Ri - s_R{i-1}) * ReLU(x - c^R_i) 
             - Σ (s_Li - s_L{i-1}) * ReLU(c^L_i - x)
        """
        out = x

        if self.no_curr_used_hinges == 0:
            return out

        a_r_all, a_l_all = self.get_coefficients()
        
        a_r = a_r_all[:self.no_curr_used_hinges]
        a_l = a_l_all[:self.no_curr_used_hinges]
        c_r = self.c_r[:self.no_curr_used_hinges]
        c_l = self.c_l[:self.no_curr_used_hinges]

        x_u = x.unsqueeze(0)

        if x.dim() == 3:
            a_r = a_r.unsqueeze(2)
            a_l = a_l.unsqueeze(2)
            c_r = c_r.unsqueeze(2)
            c_l = c_l.unsqueeze(2)

        # Right term: Adds slope changes for x > c_r
        term_r = a_r * F.relu(x_u - c_r)
        
        # Left term: Adds slope changes for x < c_l
        term_l = a_l * F.relu(c_l - x_u)

        out = out + term_r.sum(dim=0) - term_l.sum(dim=0)
        return out