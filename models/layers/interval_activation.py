import torch
from torch import nn
import torch.nn.functional as F

import numpy as np
from typing import List


class IntervalActivation(nn.Module):
    """
    IntervalActivation layer for preserving learned representations within a hypercube.

    This layer applies a Leaky ReLU activation and tracks the range of activations 
    across batches. It defines a [min, max] hypercube per neuron, which can be used 
    to enforce that activations remain within these bounds when learning new tasks.

    Attributes:
        input_shape (int): Flattened size of the input tensor.
        lower_percentile (float): Lower percentile for min bound computation.
        upper_percentile (float): Upper percentile for max bound computation.
        use_non_linear_transform (bool): Whether to apply Leaky ReLU activation.
        test_act_buffer (List[torch.Tensor]): Stores activations for percentile computation in eval mode.
        min (Optional[torch.Tensor]): Lower bound per neuron (updated via reset_range).
        max (Optional[torch.Tensor]): Upper bound per neuron (updated via reset_range).
        curr_task_last_batch (Optional[torch.Tensor]): Stores last batch activations during training.
    """

    def __init__(self,
        input_shape: tuple,
        lower_percentile: float = 0.05,
        upper_percentile: float = 0.95,
        use_non_linear_transform: bool = True
    ) -> None:
        """
        Initializes the IntervalActivation layer.

        Args:
            input_shape (Tuple[int, ...]): Shape of the input tensor.
            lower_percentile (float, optional): Lower percentile for min bound. Defaults to 0.05.
            upper_percentile (float, optional): Upper percentile for max bound. Defaults to 0.95.
            use_non_linear_transform (bool, optional): Whether to apply Leaky ReLU activation. Defaults to True.
        """

        super().__init__()
        self.init_args = (input_shape,)
        self.init_kwargs = dict(lower_percentile=lower_percentile, upper_percentile=upper_percentile)

        self.input_shape = np.prod(input_shape)
        self.lower_percentile = lower_percentile
        self.upper_percentile = upper_percentile
        self.use_non_linear_transform = use_non_linear_transform

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
        out = x.view(x.shape[0], -1)

        if self.use_non_linear_transform:
            out = F.leaky_relu(out)

        if self.training:
            self.curr_task_last_batch = out        
        else:
            self.test_act_buffer.append(out.detach().cpu().view(out.size(0), -1))

        return out