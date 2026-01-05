import torch
from torch import nn
from torch.nn import functional as F

class LearnableReLU(nn.Module):
    """
    LearnableReLU: Monotone-by-construction ReLU basis expansion
    for continual learning.

    This module represents a *learnable monotone activation function*
    constructed as a sum of shifted ReLU basis functions:

        f(z) = ∑_{i=0}^{K-1} a_i · ReLU(z - c_i)

    where:
        • z is a preactivation (typically output of a Linear layer),
        • c_i are non-decreasing hinge locations,
        • a_i are learnable coefficients.

    The coefficients are parameterized via a *cumulative-positive*
    construction that guarantees:

        ∂f(z) / ∂z ≥ 0   for all z

    i.e. the function is monotone non-decreasing with respect to z.

    This property is critical for:
        • interval arithmetic–based drift bounds,
        • exact preservation of old-task behavior,
        • safe functional expansion across tasks.

    Continual learning protocol:
    ----------------------------
    • Task 0 initializes the first hinge.
    • Each new task:
        - anchors a new hinge beyond old-task activations,
        - activates one additional basis function,
        - optionally freezes previous basis parameters.

    Capacity grows *only* by adding new hinges, preventing
    destructive interference with previously learned tasks.
    """

    def __init__(self,
        out_features: int,
        k: int) -> None:
        """
        Initialize LearnableReLU.

        Args:
            out_features (int):
                Number of output features.
            k (int):
                Maximum number of hinge basis functions
                (typically the maximum number of tasks).
        """

        super().__init__()
        
        self.k = k
        self.no_curr_used_basis_functions = 1

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Computes:
            f(x) = ∑ a_i · ReLU(x - c_i)

        using only the currently active basis functions.

        Due to the cumulative-positive construction of a_i,
        the function is guaranteed to be monotone with respect
        to its input.

        Args:
            x (Tensor):
                Input tensor of shape (batch_size, out_features).

        Returns:
            Tensor of shape (batch_size, out_features).
        """
        a = self.basis_scales()[:self.no_curr_used_basis_functions]
        c = self.cum_shifts[:self.no_curr_used_basis_functions]

        out = (a * torch.relu(x.unsqueeze(0) - c)).sum(dim=0)

        return out

        