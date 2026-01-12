import logging
from copy import deepcopy
from typing import List

import torch
import torch.nn as nn

from models.layers.interval_activation import IntervalActivation
from models.layers.learnable_relu import LearnableReLU

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class InTActPlusPlusMlpBlockRegularization(nn.Module):
    """
    InTAct++ Linear Regularization Module for Continual Learning.

    This module implements a *functional drift regularizer* that constrains how
    much a linear layer's output can change across tasks. It combines:

    - Interval Arithmetic (IA) for worst-case drift bounds
    - Residual drift bounding for discarded dimensions
    - Variance regularization for representation compactness

    The regularizer is designed to be applied as an *augmentation to the task loss*
    during training and assumes the following architectural block:

        IntervalActivation -> Linear -> LearnableReLU -> IntervalActivation -> Softmax
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
        self.interval_layer1: IntervalActivation = None
        self.interval_layer2: IntervalActivation = None
        self.curr_linear_layer: nn.Linear = None
        
        # Frozen copy of the previous layer
        self.prev_linear_layer: nn.Linear = None
        
        self.learnable_relu: LearnableReLU = None

    @torch.no_grad()
    def setup_task(
        self,
        task_id: int,
        cls_layers: List, # [Interval, Linear, LearnableReLU, Interval]
    ) -> None:
        """
        Prepare the regularizer for a new task.

        Args:
            task_id (int): Current task id.
            cls_layers (List): List of classification head layers to be regularized
        """
        self.task_id = task_id

        # 1. Map Layer References
        self.interval_layer1 = cls_layers[0]
        self.curr_linear_layer = cls_layers[1]
        self.learnable_relu = cls_layers[2]
        self.interval_layer2 = cls_layers[3]

        # 2. Deepcopy and Freeze the previous task's weights
        # We use ModuleList so they are properly moved to the correct device
        self.prev_linear_layer = deepcopy(self.curr_linear_layer).eval()
        for p in self.prev_linear_layer.parameters():
            p.requires_grad = False

        device = next(self.prev_linear_layer.parameters()).device

        if task_id == 0:
            return

        with torch.no_grad():
            # ============================================================
            # Phase 1 — Global Statistics Collection (All Tokens)
            # ============================================================
            all_inputs_fc = []
            preacts_for_hinges = []

            # We need to do it only for the first interval layer
            for x in self.interval_layer1.test_act_buffer:
                x = x.to(device)
                all_inputs_fc.append(x.detach())

                z = self.prev_linear_layer(x)
                preacts_for_hinges.append(z.detach())

            # ============================================================
            # Phase 2 — Mean Centering (The "Tightness" Trick)
            # ============================================================
            if all_inputs_fc:
                Z_all = torch.cat(all_inputs_fc, dim=0)
                input_mean = Z_all.mean(dim=0)
                self.register_buffer("input_mean_fc", input_mean)
                
                log.info(f"Task {task_id}: Global mean for fc captured. Shape: {input_mean.shape}")

            # ============================================================
            # Phase 3 — Anchor LearnableReLU Hinges
            # ============================================================
            if preacts_for_hinges:
                Z_pre = torch.cat(preacts_for_hinges, dim=0)
                
                # Anchor hinges at the 95th and 5th percentiles of old activations
                self.learnable_relu.anchor_next_shift(
                    z=Z_pre, 
                    task_id=task_id, 
                    percentile=0.95
                )
                # Enable the next basis function for the new task
                self.learnable_relu.set_no_used_basis_functions(task_id)

                if task_id > 1:
                    self.learnable_relu.freeze_basis_function(task_id-2)


            # ============================================================
            # Phase 4 — Finalize Interval Bounds
            # ============================================================
            # We trigger the reset_range for the first linear layer.
            # This computes the final [min, max] hypercube from the test_act_buffer
            self.interval_layer1.reset_range()
                
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
        for interval_layer in [self.interval_layer1, self.interval_layer2]:
            acts = interval_layer.curr_task_last_batch
            if acts is not None:
                acts_flat = acts.view(-1, acts.size(-1)) 
                var_loss += acts_flat.var(dim=0, unbiased=False).mean()


        # 2. Functional Drift Regularization (Recursive Chaining)
        if self.task_id > 0:
            # --- LAYER 1: Linear1 (fc1) ---
            delta_W = self.curr_linear_layer.weight - self.prev_linear_layer.weight
            delta_b = self.curr_linear_layer.bias - self.prev_linear_layer.bias
            
            mean_drift = delta_W @ self.input_mean_fc.to(x.device)
            effective_bias = delta_b + mean_drift

            lb = self.interval_layer1.min.to(x.device)
            ub = self.interval_layer1.max.to(x.device)
            
            dW1_pos, dW1_neg = torch.relu(delta_W), torch.relu(-delta_W)

            drift_low = dW1_pos @ lb - dW1_neg @ ub + effective_bias
            drift_up  = dW1_pos @ ub - dW1_neg @ lb + effective_bias
            drift_loss += (drift_low.pow(2).mean() + drift_up.pow(2).mean())


        return loss + (self.lambda_var * var_loss) + \
                      (self.lambda_drift * drift_loss)