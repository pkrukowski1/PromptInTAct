import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnableReLU(nn.Module):
    """
    Mechanism:
    1. Learns 'theta' (unbounded parameter).
    2. Bounds it: s = max_dev * tanh(theta).
    3. Telescopes it: w = s_curr - s_prev.
    """

    def __init__(self, out_features: int, k: int, base_function: nn.Module = nn.Identity()) -> None:
        super().__init__()
        self.k = k
        self.num_active_hinges = 0
        self.base_function = base_function

        # Unbounded parameters (Thetas)
        self.theta_r = nn.ParameterList(
            nn.Parameter(torch.zeros(1, out_features)) for _ in range(k - 1)
        )
        self.theta_l = nn.ParameterList(
            nn.Parameter(torch.zeros(1, out_features)) for _ in range(k - 1)
        )

        self.register_buffer("c_r", torch.ones(k - 1, 1, out_features) * 9999.0)
        self.register_buffer("c_l", torch.ones(k - 1, 1, out_features) * -9999.0)
        
        self.register_buffer("global_max", torch.ones(1, out_features) * -9999.0)
        self.register_buffer("global_min", torch.ones(1, out_features) * 9999.0)

    def set_no_used_basis_functions(self, task_id: int) -> None:
        self.num_active_hinges = task_id

    def freeze_basis_function(self, idx: int) -> None:
        if idx < len(self.theta_r):
            self.theta_r[idx].requires_grad_(False)
            self.theta_l[idx].requires_grad_(False)

    @torch.no_grad()
    def anchor_next_shift(
        self,
        z: torch.Tensor,
        task_id: int,
        percentile_high: float = 0.99,
        percentile_low: float = 0.01,
    ) -> None:
        z_cpu = z.detach().cpu()
        if z_cpu.dim() == 3: 
            z_cpu = z_cpu.reshape(-1, z_cpu.size(-1))

        curr_max = torch.quantile(z_cpu, percentile_high, dim=0, keepdim=True)
        curr_min = torch.quantile(z_cpu, percentile_low, dim=0, keepdim=True)
        
        device = self.c_r.device
        curr_max = curr_max.to(device)
        curr_min = curr_min.to(device)
        
        idx = task_id - 1
        if idx >= 0 and idx < len(self.c_r):
            if task_id == 1:
                self.c_r[idx] = curr_max
                self.c_l[idx] = curr_min
                self.global_max = curr_max
                self.global_min = curr_min
            else:
                self.c_r[idx] = self.global_max
                self.c_l[idx] = self.global_min
                self.global_max = torch.maximum(self.global_max, curr_max)
                self.global_min = torch.minimum(self.global_min, curr_min)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_function(x)

        if self.num_active_hinges == 0:
            return base

        c_r_curr = self.c_r[:self.num_active_hinges]
        c_l_curr = self.c_l[:self.num_active_hinges]

        theta_r_stack = torch.stack([self.theta_r[i] for i in range(self.num_active_hinges)])
        theta_l_stack = torch.stack([self.theta_l[i] for i in range(self.num_active_hinges)])
        
        s_r = -0.5 * (torch.tanh(theta_r_stack) + 1.0)
        s_l = -0.5 * (torch.tanh(theta_l_stack) + 1.0)
        
        # Telescoping weights
        zeros = torch.zeros_like(s_r[0:1]) 
        s_r_shifted = torch.cat([zeros, s_r[:-1]], dim=0)
        w_r_stack = s_r - s_r_shifted
        
        s_l_shifted = torch.cat([zeros, s_l[:-1]], dim=0)
        w_l_stack = s_l - s_l_shifted

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
        
        return base + correction_r.sum(dim=0) - correction_l.sum(dim=0)