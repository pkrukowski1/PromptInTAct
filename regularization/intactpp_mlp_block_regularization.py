import logging
from copy import deepcopy
from typing import List

import torch
import torch.nn as nn

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class InTActPlusPlusMlpBlockRegularization(nn.Module):
    """
    InTAct++ Linear Regularization Module for Continual Learning.

    This module implements a *functional drift regularizer* that constrains how
    much a linear layer's output can change across tasks. It combines:

    - Interval Arithmetic (IA) for worst-case drift bounds
    - SVD-based low-dimensional subspace projection
    - Residual drift bounding for discarded dimensions
    - Variance regularization for representation compactness
    - Slope regularization for LearnableReLU stability

    The regularizer is designed to be applied as an *augmentation to the task loss*
    during training and assumes the following architectural block:

        IntervalActivation -> Linear -> LearnableReLU
    """
    def __init__(self,
            lambda_var: float = 0.01,
            lambda_slope_reg: float = 0.01,
            lambda_drift: float = 1.0,
            reduced_dim: int = 50,
        ) -> None:
        """
        Initialize the InTAct++ regularizer.

        Args:
            lambda_var (float): Weight for activation variance regularization.
            lambda_slope_reg (float): Weight for LearnableReLU slope regularization.
            lambda_drift (float): Weight for functional drift penalty.
            reduced_dim (int): Dimensionality of the SVD projection subspace.
        """
        
        super().__init__()
        self.task_id = None
        log.info(
            f"InTAct++ initialized with "
            f"lambda_var={lambda_var}, "
            f"lambda_slope_reg={lambda_slope_reg}, "
            f"lambda_drift={lambda_drift}"
        )

        self.task_id = None
        self.lambda_var = lambda_var
        self.lambda_slope_reg = lambda_slope_reg
        self.lambda_drift = lambda_drift

        # References to current layers
        self.interval_layers = []
        self.curr_linear_layers = []
        
        # Frozen copies of previous layers
        self.prev_linear_layers = nn.ModuleList()
        
        self.learnable_relu = None

    @torch.no_grad()
    def setup_task(
        self,
        task_id: int,
        mlp_layers: List, # [Interval1, Linear1, LearnableReLU, Interval2, Linear2]
    ) -> None:
        """
        Prepare the regularizer for a new task by anchoring hinges,
        calculating global token means, and freezing old weights.

        Args:
            task_id (int): Current task id.
            mlp_layers (List): List of layers from a ViT block to be regularized
        """
        self.task_id = task_id

        if task_id == 0:
            return

        # 1. Map Layer References
        # interval_layers[0] guards linear_layers[0] (fc1)
        # interval_layers[1] guards linear_layers[1] (fc2)
        self.interval_layers = [mlp_layers[0], mlp_layers[3]]
        self.curr_linear_layers = [mlp_layers[1], mlp_layers[4]]
        self.learnable_relu = mlp_layers[2]

        # 2. Deepcopy and Freeze the previous task's weights
        # We use ModuleList so they are properly moved to the correct device
        self.prev_linear_layers = nn.ModuleList([
            deepcopy(self.curr_linear_layers[0]).eval(),
            deepcopy(self.curr_linear_layers[1]).eval()
        ])
        
        for layer in self.prev_linear_layers:
            for p in layer.parameters():
                p.requires_grad = False

        self.interval_act_layer1 = self.interval_layers[0]
        self.curr_linear_layer1  = self.curr_linear_layers[0]
        self.prev_linear_layer1  = self.prev_linear_layers[0]

        self.interval_act_layer2 = self.interval_layers[1]
        self.curr_linear_layer2  = self.curr_linear_layers[1]
        self.prev_linear_layer2  = self.prev_linear_layers[1]

        device = next(self.curr_linear_layers[0].parameters()).device

        # ============================================================
        # Phase 1 — Global Statistics Collection (All Tokens)
        # ============================================================
        all_inputs_fc1 = []
        preacts_for_hinges = []

        for x in self.interval_layers[0].test_act_buffer:
            x = x.to(device)
            all_inputs_fc1.append(x.detach())

            z = self.prev_linear_layers[0](x)
            preacts_for_hinges.append(z.detach())

        # ============================================================
        # Phase 2 — Mean Centering (The "Tightness" Trick)
        # ============================================================
        if all_inputs_fc1:
            Z_all = torch.cat(all_inputs_fc1, dim=0)
            
            # Calculate the global mean of ALL tokens in the dataset
            # This is used in 'forward' to center the interval expansion
            input_mean = Z_all.mean(dim=0)
            self.register_buffer("input_mean_fc1", input_mean)
            
            log.info(f"Task {task_id}: Global mean for fc1 captured. Shape: {input_mean.shape}")

        # ============================================================
        # Phase 3 — Anchor LearnableReLU Hinges
        # ============================================================
        if preacts_for_hinges:
            Z_pre = torch.cat(preacts_for_hinges, dim=0)
            
            # Anchor hinges at the 95th percentile of old activations
            self.learnable_relu.anchor_next_shift(
                z=Z_pre, 
                task_id=task_id, 
                percentile=0.95
            )
            # Enable the next basis function for the new task
            self.learnable_relu.set_no_used_basis_functions(task_id + 1)

        # ============================================================
        # Phase 4 — Finalize Interval Bounds
        # ============================================================
        # We trigger the reset_range for both Interval layers.
        # This computes the final [min, max] hypercube from the test_act_buffer
        for layer in self.interval_layers:
            layer.reset_range()
            
        log.info(f"Task {task_id} setup complete. Regularizing against Task {task_id-1}.")

                    
    def forward(self, x: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        """
        Augment task loss with InTAct++ regularization terms.

        Regularization components:
            - Activation variance minimization
            - LearnableReLU slope regularization
            - Functional drift penalty (interval-based)

        Args:
            x (torch.Tensor): Input batch.
            loss (torch.Tensor): Task loss.

        Returns:
            torch.Tensor: Total loss.
        """

        var_loss = torch.tensor(0.0, device=x.device)
        drift_loss = torch.tensor(0.0, device=x.device)

        # 1. Variance Regularization
        for interval_layer in self.interval_layers:
            acts = interval_layer.curr_task_last_batch
            if acts is not None:
                # Sequence-aware: treat every token as a sample
                acts_flat = acts.view(-1, acts.size(-1)) 
                var_loss += acts_flat.var(dim=0, unbiased=False).mean()

        # 2. Slope Regularization
        slope = self.learnable_relu.raw_scales[self.task_id]
        slope_loss = slope.pow(2).mean()

        # 3. Pure Interval Drift Regularization
        if self.task_id > 0:
            delta_W = self.curr_linear_layer1.weight - self.prev_linear_layer1.weight
            delta_b = self.curr_linear_layer1.bias - self.prev_linear_layer1.bias
            
            # This calculates how much the output 'jumps' just because of the weights changing
            # relative to the average input signal.
            mean_drift = delta_W @ self.input_mean.to(x.device)
            effective_bias = delta_b + mean_drift

            lb = self.interval_act_layer1.min.to(x.device)
            ub = self.interval_act_layer1.max.to(x.device)
            
            dW_pos = torch.relu(delta_W)
            dW_neg = torch.relu(-delta_W)

            drift_low = dW_pos @ lb - dW_neg @ ub + effective_bias
            drift_up  = dW_pos @ ub - dW_neg @ lb + effective_bias

            drift_loss += (drift_low.pow(2).mean() + drift_up.pow(2).mean())

        return loss + (self.lambda_var * var_loss) + \
                      (self.lambda_slope_reg * slope_loss) + \
                      (self.lambda_drift * drift_loss)