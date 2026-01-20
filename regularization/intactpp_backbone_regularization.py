import logging

import torch
import torch.nn as nn

from models.layers.interval_activation import IntervalActivation
from models.layers.learnable_relu import LearnableReLU

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class InTActPlusPlusBackboneRegularization(nn.Module):
    """
    InTAct++ Regularization Module for ViT Backbone Continual Learning.

    This module implements a *functional drift regularizer* designed for the 
    activation layers within a Vision Transformer (ViT) backbone. It constrains 
    how much the activation distributions can shift across tasks to prevent 
    catastrophic forgetting while allowing plasticity via learnable activations.

    It combines:
    - Interval Arithmetic-based drift bounds (geometric alignment)
    - Variance regularization for representation compactness
    - Management of LearnableReLU basis functions

    The regularizer is designed to be applied as an *augmentation to the task loss*
    during training. It assumes the following position within a Transformer MLP block:

    [... -> Linear -> IntervalActivation -> LearnableReLU -> Linear -> ...]
    """
    def __init__(self,
            lambda_var: float = 0.01,
        ) -> None:
        """
        Initialize the InTAct++ regularizer.

        Args:
            lambda_var (float): Weight for activation variance regularization.
        """
        
        super().__init__()
        self.task_id = None
        log.info(
            f"InTAct++ for Backbone regularization initialized with "
            f"lambda_var={lambda_var}"
        )

        self.task_id = None
        self.lambda_var = lambda_var

        self.interval_layer: IntervalActivation = None
        self.learnable_relu: LearnableReLU = None
        
        
    @torch.no_grad()
    def setup_task(
        self,
        task_id: int,
        interval_layer: IntervalActivation,
        learnable_relu: LearnableReLU,
    ) -> None:
        """
        Prepare the regularizer for a new task.

        Collects statistics from the previous task to anchor the LearnableReLU
        hinges and establishes the interval bounds for drift detection.

        Args:
            task_id (int): Current task ID.
            interval_layer (IntervalActivation): The interval tracking layer preceding the activation.
            learnable_relu (LearnableReLU): The learnable activation function module.
        """
        self.task_id = task_id

        # 1. Map Layer References
        self.interval_layer = interval_layer
        self.learnable_relu = learnable_relu

        assert isinstance(self.interval_layer, IntervalActivation)
        assert isinstance(self.learnable_relu, LearnableReLU)

        device = next(self.learnable_relu.parameters()).device

        if task_id == 0:
            return

        with torch.no_grad():
            # ============================================================
            # Phase 1 — Global Statistics Collection (All Tokens)
            # ============================================================
            preacts_for_hinges = []

            for x in self.interval_layer.test_act_buffer:
                x = x.to(device)
                preacts_for_hinges.append(x.detach())

            # ============================================================
            # Phase 2 — Anchor LearnableReLU Hinges
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
            # Phase 3 — Finalize Interval Bounds
            # ============================================================
            self.interval_layer.reset_range()
                
            log.info(f"Task {task_id} setup complete. Regularizing against Task {task_id-1}.")

                    
    def forward(self, x: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        """
        Augment task loss with InTAct++ regularization terms.

        Calculates penalties for:
        1. **Variance Loss**: Encourages compact representations within the current batch.
        2. **Alignment Loss**: Penalizes the center of mass shift if the current batch's 
           activation range does not overlap with the previous task's learned hypercube.

        Args:
            x (torch.Tensor): Input batch (not explicitly used as acts are retrieved from layer).
            loss (torch.Tensor): The original task loss (e.g., CrossEntropy).

        Returns:
            torch.Tensor: The total loss (Task Loss + Regularization).
        """
        
        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        var_loss = zero.clone()
        align_repr_loss = zero.clone()

        acts = self.interval_layer.curr_task_last_batch
        acts_flat = acts.view(acts.size(0), -1)
        batch_var = acts_flat.var(dim=0, unbiased=False).mean()
        var_loss += batch_var
        
        if self.task_id > 0:
            lb = self.interval_layer.min.to(x.device)
            ub = self.interval_layer.max.to(x.device)
            prev_center = (ub + lb) / 2.0
            prev_radii  = (ub - lb) / 2.0

            lb_prev_hypercube = prev_center - prev_radii
            ub_prev_hypercube = prev_center + prev_radii

            new_lb, _ = acts_flat.min(dim=0)
            new_ub, _ = acts_flat.max(dim=0)

            non_overlap_mask = (new_lb > ub_prev_hypercube) | (new_ub < lb_prev_hypercube)
            new_center = (new_ub + new_lb) / 2.0

            center_loss = torch.norm(new_center[non_overlap_mask] - prev_center[non_overlap_mask], p=2)

            align_repr_loss += center_loss / (prev_radii.mean() + 1e-8)

        return loss + \
                self.lambda_var * var_loss + \
                align_repr_loss
            