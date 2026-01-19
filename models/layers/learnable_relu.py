import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnableReLU(nn.Module):
    """
    LearnableReLU with guaranteed sublinear growth:
        0 <= f'(x) <= 1

    This prevents amplification and makes IA drift-safe by construction.
    """

    def __init__(self, out_features: int, k: int) -> None:
        super().__init__()

        self.k = k
        self.out_features = out_features
        self.no_curr_used_hinges = 0

        # Raw decay parameters
        self.raw_decay_r = nn.ParameterList(
            nn.Parameter(torch.ones(1, out_features) * -5.0) for _ in range(k - 1)
        )
        self.raw_decay_l = nn.ParameterList(
            nn.Parameter(torch.ones(1, out_features) * -5.0) for _ in range(k - 1)
        )

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
        self.raw_decay_r[idx].requires_grad_(False)
        self.raw_decay_l[idx].requires_grad_(False)
        

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

    def get_coefficients(self) -> None:
        """
        Calculates coefficients a_r, a_l such that:
        1. All a_i <= 0 (Sublinear growth)
        2. 1 + Sum(a_i) >= 0.01 (Strict Monotonicity)
        """
        raw_decays_r = [F.softplus(p) for p in self.raw_decay_r]
        raw_decays_l = [F.softplus(p) for p in self.raw_decay_l]

        if not raw_decays_r:
            return None, None

        D_R = torch.stack(raw_decays_r, dim=0) 
        D_L = torch.stack(raw_decays_l, dim=0)

        # 2. Compute Normalization
        sum_R = D_R.sum(dim=0, keepdim=True)
        sum_L = D_L.sum(dim=0, keepdim=True)
        
        limit = 0.99
        factor_R = torch.maximum(torch.ones_like(sum_R), sum_R / limit)
        factor_L = torch.maximum(torch.ones_like(sum_L), sum_L / limit)
        
        final_D_R = D_R / factor_R
        final_D_L = D_L / factor_L

        # 3. Return negative coefficients
        return -final_D_R, -final_D_L
    
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

        term_r = a_r * F.relu(x_u - c_r)        
        term_l = a_l * F.relu(c_l - x_u)

        out = out + term_r.sum(dim=0) - term_l.sum(dim=0)
        return out