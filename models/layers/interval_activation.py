import torch
from torch import nn
import torch.nn.functional as F

import numpy as np

class IntervalActivation(nn.Module):
    """
    IntervalActivation layer for preserving learned representations within a hypercube.

    This layer applies a Leaky ReLU activation and tracks the range of activations 
    across batches. It defines a [min, max] hypercube per neuron, which can be used 
    to enforce that activations remain within these bounds when learning new tasks.

    Attributes:
        lower_percentile (float): Lower percentile for min bound computation.
        upper_percentile (float): Upper percentile for max bound computation.
        use_non_linear_transform (bool): Whether to apply Leaky ReLU activation.
        test_act_buffer (List[torch.Tensor]): Stores activations for percentile computation in eval mode.
        min (Optional[torch.Tensor]): Lower bound per neuron (updated via reset_range).
        max (Optional[torch.Tensor]): Upper bound per neuron (updated via reset_range).
        curr_task_last_batch (Optional[torch.Tensor]): Stores last batch activations during training.
        maintain_test_act_buffer (bool, optional): If True, test activation buffer is maintained to calculate bounds.
    """

    def __init__(self,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95,
        use_non_linear_transform: bool = False,
        maintain_test_act_buffer: bool = True
    ) -> None:
        """
        Initializes the IntervalActivation layer.

        Args:
            lower_percentile (float, optional): Lower percentile for min bound. Defaults to 0.05.
            upper_percentile (float, optional): Upper percentile for max bound. Defaults to 0.95.
            use_non_linear_transform (bool, optional): Whether to apply Leaky ReLU activation. Defaults to True.
            maintain_test_act_buffer (bool, optional): If True, test activation buffer is maintained to calculate bounds.
        """

        super().__init__()
        self.init_kwargs = dict(lower_percentile=lower_percentile, upper_percentile=upper_percentile)

        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.use_non_linear_transform = use_non_linear_transform
        self.maintain_test_act_buffer = maintain_test_act_buffer

        self.min = None
        self.max = None

        self.curr_task_last_batch = None
        self.test_act_buffer = []

    def reset_range(self) -> None:
        """
        Updates the [min, max] hypercube for each neuron using stored activations.

        Steps:
            1. Concatenate stored activations in test_act_buffer.
            2. Sort activations and select lower and upper percentiles.
            3. Update self.min and self.max by element-wise min/max.
            4. Clear the test_act_buffer.
        """
        
        if len(self.test_act_buffer) == 0:
            return

        activations = torch.cat(self.test_act_buffer, dim=0)
        sorted_buf, _ = torch.sort(activations, dim=0)
      
        n = sorted_buf.size(0)
        if n == 0:
            return

        l_idx = int(np.clip(int(n * self.lower_percentile), 0, n - 1))
        u_idx = int(np.clip(int(n * self.upper_percentile), 0, n - 1))

        min_vals = sorted_buf[l_idx]   # shape (d,)
        max_vals = sorted_buf[u_idx]   # shape (d,)
        
        if self.min is None or self.max is None:
            self.min = min_vals.clone()
            self.max = max_vals.clone()
        else:
            self.min = torch.minimum(self.min, min_vals)
            self.max = torch.maximum(self.max, max_vals)

        self.test_act_buffer = []


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes activations for input x.

        During training:
            - Stores batch activations in curr_task_last_batch.
        
        During evaluation:
            - Stores activations in test_act_buffer for later percentile computation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, ...).

        Returns:
            torch.Tensor: Activated tensor of shape (batch, input_shape).
        """
        out = x 
        if self.use_non_linear_transform:
            out = F.leaky_relu(out)

        # Extract Representation for Regularization (CLS Token Logic)
        # We only want to store/regularize the CLS token to save memory 
        # and focus on semantic drift rather than spatial noise.
        if out.dim() == 3:
            repr_for_stats = out[:, 0, :] 
        else:
            repr_for_stats = out

        if self.training:
            self.curr_task_last_batch = repr_for_stats
        elif self.maintain_test_act_buffer:
            self.test_act_buffer.append(repr_for_stats.detach().cpu())

        return out