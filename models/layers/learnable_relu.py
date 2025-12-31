import torch
from torch import nn
from torch.nn import functional as F

import math

class LearnableReLU(nn.Module):

    def __init__(self,
        in_features: int,
        out_features: int,
        k: int) -> None:
        """
        Linear layer augmented with task-wise ReLU hinge basis functions
        with *monotone-by-construction* derivatives, designed for
        continual learning.

        This module implements a function of the form:

            f(x) = ∑_{i=0}^{T-1} a_i · ReLU(Wx + b - c_i)

        where:
        - W, b define a shared linear preactivation z = Wx + b,
        - c_i are non-decreasing hinge locations (shifts),
        - a_i are learnable basis coefficients derived from a
          cumulative-positive parameterization.

        Crucially, the coefficients a_i are constructed such that
        **all partial sums of coefficients are non-negative**, which
        guarantees that:

            ∂f(z) / ∂z ≥ 0   for all z

        i.e. the function is *monotone non-decreasing* with respect to
        the preactivation z.

        This property enables:
        - exact invariance of old tasks under interval-preserving updates,
        - analytical regularization using interval arithmetic,
        - safe expansion of the function by adding new hinge basis
          functions without breaking previously learned behavior.

        Continual learning protocol:
        ----------------------------
        • Task 0 initializes the first hinge.
        • Each new task:
            - freezes previously learned coefficients,
            - anchors a new hinge location beyond old-task activations,
            - activates one additional basis function.

        The representation capacity grows *only* by adding new hinges,
        while the monotonicity constraint prevents destructive interference.

        Args:
            in_features (int):
                Number of input features.
            out_features (int):
                Number of output features.
            k (int):
                Maximum number of hinge basis functions
                (typically equal to the maximum number of tasks).
        """

        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.k = k

        self.no_curr_used_basis_functions = 1

        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=True)
        self.bias = nn.Parameter(torch.empty(1, out_features), requires_grad=True)

        # Unconstrained parameters
        # These parameters are NOT the actual coefficients a_i.
        # Instead, they are transformed via a cumulative-softplus
        # construction to guarantee monotone derivatives.
        self.raw_scales = nn.ParameterList(
            nn.Parameter(torch.zeros(1, out_features)) for _ in range(k)
        )

        # Non-trainable shifts
        self.register_buffer(
            "cum_shifts",
            torch.zeros(k, 1, out_features)
        )

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize layer parameters.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)

    def cumulative_scales(self) -> torch.Tensor:
        """
        Compute cumulative-positive scale values.

        Returns:
            Tensor S of shape (k, 1, out_features) such that:
                S_i = softplus(raw_scales[i]) > 0

        These values represent cumulative sums of basis coefficients
        and are guaranteed to be non-negative.
        """
        raw = torch.stack(list(self.raw_scales), dim=0)
        return F.softplus(raw)
    
    def basis_scales(self) -> torch.Tensor:
        """
        Compute actual basis coefficients a_i.

        The coefficients are defined as:
            a_0 = S_0
            a_i = S_i - S_{i-1}   for i > 0

        This construction guarantees:
            ∑_{j=0}^i a_j = S_i ≥ 0

        Individual coefficients a_i may be negative, but all partial
        sums are non-negative, ensuring a non-negative derivative of
        the overall function.

        Returns:
            Tensor of shape (k, 1, out_features) containing a_i.
        """
        S = self.cumulative_scales()
        a = S.clone()
        a[1:] = S[1:] - S[:-1]
        return a

    
    def set_no_used_basis_functions(self, value: int) -> None:
        """
        Set the number of currently active basis functions.

        This method is typically called when a new task is introduced
        in a continual learning setting, enabling an additional
        ReLU basis function while keeping previously learned basis
        functions unchanged.

        Args:
            value (int): Number of basis functions to be used.
        """
        self.no_curr_used_basis_functions = value
    
    def freeze_basis_function(self, idx: int) -> None:
        """
        Freeze a learnable ReLU basis function.

        This method disables gradient updates for the scale
        parameters associated with a specific basis function. It is
        typically used in a continual learning setting to prevent
        modification of basis functions learned for previous tasks
        while allowing new basis functions to be trained.

        Args:
            idx (int): Index of the basis function to freeze.
        """
        self.raw_scales[idx].requires_grad_(False)
    
    @torch.no_grad()
    def anchor_next_shift(self, z: torch.Tensor, task_id: int, percentile: float=0.95) -> None:
        """
        Anchor the hinge location for a new task.

        The new hinge c_task_id is placed at a high percentile of the
        preactivation distribution of the completed task, ensuring that:

            ReLU(z - c_task_id) = 0   for (almost) all old-task data

        Additionally, hinge locations are enforced to be monotone
        non-decreasing:

            c_0 ≤ c_1 ≤ ... ≤ c_T

        Args:
            z (Tensor):
                Collected preactivations of shape (N, out_features).
            task_id (int):
                Index of the newly introduced task.
            percentile (float):
                Upper percentile used to anchor the hinge.
        """
        P = torch.quantile(z, percentile, dim=0, keepdim=True)
        if task_id == 0:
            self.cum_shifts[0] = P
        else:
            self.cum_shifts[task_id] = torch.maximum(
                P, self.cum_shifts[task_id - 1]
            )

    def _min_derivative_interval(self, x_min: torch.Tensor, x_max: torch.Tensor) -> torch.Tensor:
        """
        Compute the worst-case (minimum) derivative of the function
        over an input hypercube.

        This method uses interval arithmetic to bound the preactivation
        z = Wx + b over x ∈ [x_min, x_max], and evaluates the derivative
        at the worst-case point.

        The returned value lower-bounds:

            ∂f(z) / ∂z

        over the entire input region. A non-negative result certifies
        that the function is monotone over the interval, which is
        sufficient to guarantee invariance of old-task behavior.

        Args:
            x_min (Tensor):
                Lower corner of the input hypercube.
            x_max (Tensor):
                Upper corner of the input hypercube.

        Returns:
            Tensor of shape (batch_size, out_features) containing the
            minimum derivative values.
        """
        x_c = 0.5 * (x_min + x_max)
        x_r = 0.5 * (x_max - x_min)

        mu = F.linear(x_c, self.weight, self.bias)
        rad = F.linear(x_r, self.weight.abs())

        z_wc = mu - rad  # worst-case

        a = self.basis_scales()[:self.no_curr_used_basis_functions]
        c = self.cum_shifts[:self.no_curr_used_basis_functions]

        deriv = torch.zeros_like(z_wc)
        for ai, ci in zip(a, c):
            deriv += ai * (z_wc > ci).float()

        return deriv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Computes:
            f(x) = ∑ a_i · ReLU(Wx + b - c_i)

        using only the currently active basis functions.

        Thanks to the cumulative-positive construction of a_i, this
        function is guaranteed to be monotone with respect to the
        preactivation z = Wx + b, regardless of the sign of individual
        coefficients.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features).
            regularization_mode (bool): If True, the function is used in regularization mode 
                (without the last basis function).

        Returns:
            Tensor of shape (batch_size, out_features).
        """
        z = F.linear(x, self.weight, self.bias)
        a = self.basis_scales()[:self.no_curr_used_basis_functions]
        c = self.cum_shifts[:self.no_curr_used_basis_functions]

        out = (a * torch.relu(z.unsqueeze(0) - c)).sum(dim=0)

        return out

        