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

    The regularizer is designed to be applied as an *augmentation to the task loss*
    during training and assumes the following architectural block:

        IntervalActivation -> Linear -> LearnableReLU -> IntervalActivation -> Linear
    """
    def __init__(self,
            lambda_var: float = 0.01,
            lambda_drift: float = 1.0,
        ) -> None:
        """
        Initialize the InTAct++ regularizer.

        Args:
            lambda_var (float): Weight for activation variance regularization.
            lambda_drift (float): Weight for functional drift penalty.
        """
        
        super().__init__()
        self.task_id = None
        log.info(
            f"InTAct++ for MLP block regularization initialized with "
            f"lambda_var={lambda_var}, "
            f"lambda_drift={lambda_drift}"
        )

        self.task_id = None
        self.lambda_var = lambda_var
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

        if task_id == 0:
            return

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
        # Phase 2 — Anchor LearnableReLU Hinges
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
        # Phase 3 — Finalize Interval Bounds
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
            - Functional drift penalty (interval-based)

        Args:
            x (torch.Tensor): Input batch.
            loss (torch.Tensor): Task loss.

        Returns:
            torch.Tensor: Total loss.
        """

        var_loss = torch.tensor(0.0, device=x.device)
        drift_loss = torch.tensor(0.0, device=x.device)

        # 1. Variance Regularization (Compactness)
        for interval_layer in self.interval_layers:
            acts = interval_layer.curr_task_last_batch
            if acts is not None:
                acts_flat = acts.view(acts.size(0), -1)
                batch_var = acts_flat.var(dim=0, unbiased=False).mean()
                var_loss += batch_var


        # 2. Functional Drift Regularization (Recursive Chaining)
        if self.task_id > 0:
            # --- LAYER 1: Linear1 (fc1) ---
            delta_W1 = self.curr_linear_layer1.weight - self.prev_linear_layer1.weight
            delta_b1 = self.curr_linear_layer1.bias - self.prev_linear_layer1.bias

            lb1 = self.interval_act_layer1.min.to(x.device)
            ub1 = self.interval_act_layer1.max.to(x.device)
            
            dW1_pos, dW1_neg = torch.relu(delta_W1), torch.relu(-delta_W1)

            drift_low1 = dW1_pos @ lb1 - dW1_neg @ ub1 + delta_b1
            drift_up1  = dW1_pos @ ub1 - dW1_neg @ lb1 + delta_b1
            drift_loss += (drift_low1.pow(2).mean() + drift_up1.pow(2).mean())

            # --- LAYER 2: Linear2 (fc2) ---
            delta_W2 = self.curr_linear_layer2.weight - self.prev_linear_layer2.weight
            delta_b2 = self.curr_linear_layer2.bias - self.prev_linear_layer2.bias

            # Interval bounds for the latent space AFTER LearnableReLU
            lb2 = self.interval_act_layer2.min.to(x.device)
            ub2 = self.interval_act_layer2.max.to(x.device)
            
            dW2_pos, dW2_neg = torch.relu(delta_W2), torch.relu(-delta_W2)

            drift_low2 = dW2_pos @ lb2 - dW2_neg @ ub2 + delta_b2
            drift_up2  = dW2_pos @ ub2 - dW2_neg @ lb2 + delta_b2
            drift_loss += (drift_low2.pow(2).mean() + drift_up2.pow(2).mean())

        return loss + (self.lambda_var * var_loss) + \
                      (self.lambda_drift * drift_loss)