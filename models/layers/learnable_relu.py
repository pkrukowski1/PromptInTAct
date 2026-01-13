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
        self.no_curr_used_hinges = 0

        self.raw_sums = nn.ParameterList(
            nn.Parameter(torch.zeros(1, out_features)) for _ in range(k - 1)
        )
        self.a_r_free = nn.ParameterList(
            nn.Parameter(torch.zeros(1, out_features)) for _ in range(k - 1)
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
        self.raw_sums[idx].requires_grad_(False)
        self.a_r_free[idx].requires_grad_(False)

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

    def get_task_coefficients(self):
        """
        Constructs coefficients guaranteeing monotonicity.

        Returns:
            a_r, a_l : shape [k-1, 1, D]
        """
        # S: [k-1, 1, D]
        raw_s = torch.stack(list(self.raw_sums), dim=0)
        S = F.softplus(raw_s)

        # To calculate delta_S_i = S_i - S_{i-1}:
        # S_prev for the first hinge is 1.0 (Identity)
        S_shifted = torch.cat([torch.ones_like(S[:1]), S[:-1]], dim=0)
        delta_S = S - S_shifted # [k-1, 1, D]

        a_r = torch.stack(list(self.a_r_free), dim=0)
        # Requirement: a_r - a_l = delta_S => a_l = a_r - delta_S
        a_l = a_r - delta_S

        return a_r, a_l

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        f(x) = x
             + Σ a^R_i ReLU(x - c^R_i)
             - Σ a^L_i ReLU(c^L_i - x)
        """

        out = x

        if self.no_curr_used_hinges == 0:
            return out

        a_r_all, a_l_all = self.get_task_coefficients()

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
