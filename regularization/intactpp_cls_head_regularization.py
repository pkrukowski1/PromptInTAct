import logging
from copy import deepcopy
from typing import List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from learners.prompt import DualPrompt, L2P, CODAPrompt
from models.layers.interval_activation import IntervalActivation
from models.layers.learnable_relu import LearnableReLU

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class InTActPlusPlusClsHeadRegularization(nn.Module):
    """
    InTAct++ Linear Regularization Module (Hinge-Endpoint Version).

    This module enforces stability by 'pinning' the decision boundary at the 
    edges of previous tasks' feature distributions.

    Architectural Assumption (The 'Gatekeeper' Pattern):
    [Input/Backbone] -> IntervalActivation -> LearnableReLU -> LinearHead -> [Logits]

    Mechanism:
    1. IntervalActivation: Captures statistics of incoming backbone features.
    2. LearnableReLU: Defines 'Anchors' (Safe Zones) for old tasks.
    3. Hinge Regularization: Forces the LinearHead to output consistent 
       predictions at these Anchor points.
    """
    def __init__(self,
            lambda_var: float = 0.01,
            lambda_drift: float = 1.0,
            lambda_feat: float = 1.0,
        ) -> None:
        """
        Initialize the InTAct++ regularizer.

        Args:
            lambda_var (float): Weight for activation variance (Compactness).
            lambda_drift (float): Weight for Hinge Endpoint stability (The 'Pinning' loss).
            lambda_feat (float): Weight for Backbone Feature Drift (Distillation).
        """
        
        super().__init__()
        self.task_id = None
        log.info(
            f"InTAct++ Hinge-Regularizer initialized with "
            f"lambda_var={lambda_var}, "
            f"lambda_feat={lambda_feat}, "
            f"lambda_drift={lambda_drift} (Endpoint Pinning)"
        )

        self.lambda_var = lambda_var
        self.lambda_drift = lambda_drift
        self.lambda_feat = lambda_feat

        # Layer References
        self.interval_layer: IntervalActivation = None
        self.learnable_relu: LearnableReLU = None
        self.curr_linear_layer: nn.Linear = None
        
        # Frozen copy of the previous linear head
        self.prev_linear_layer: nn.Linear = None

        self.prompt: Union[CODAPrompt, L2P, DualPrompt] = None
        self.old_prompt: Union[CODAPrompt, L2P, DualPrompt] = None
        self.feature_extractor: nn.Sequential = None
        
    @torch.no_grad()
    def setup_task(
        self,
        task_id: int,
        cls_layers: List,  # EXPECTED: [IntervalActivation, LearnableReLU, nn.Linear]
        feature_extractor: nn.Sequential,
        prompt: Union[CODAPrompt, L2P, DualPrompt]
    ) -> None:
        """
        Prepare the regularizer for a new task.

        Args:
            task_id (int): Current task id.
            cls_layers (List): The bottleneck layers in order: 
                               [Interval, Gate/ReLU, LinearHead]
        """
        self.task_id = task_id

        # 1. Map Layer References
        self.interval_layer = cls_layers[0]
        self.learnable_relu = cls_layers[1]
        self.curr_linear_layer = cls_layers[2]
        
        assert isinstance(self.interval_layer, IntervalActivation)
        assert isinstance(self.learnable_relu, LearnableReLU)
        assert isinstance(self.curr_linear_layer, nn.Linear)

        self.feature_extractor = feature_extractor
        self.prompt = prompt

        # 2. Freeze previous layers
        self.prev_linear_layer = deepcopy(self.curr_linear_layer).eval()
        for p in self.prev_linear_layer.parameters():
            p.requires_grad = False

        self.old_prompt = deepcopy(self.prompt)
        for p in self.old_prompt.parameters():
            p.requires_grad = False

        device = next(self.prev_linear_layer.parameters()).device

        if task_id == 0:
            return

        with torch.no_grad():
            # ============================================================
            # Phase 1 — Global Statistics Collection
            # ============================================================
            # We collect raw features from the Interval layer (Output of Backbone)
            features_for_anchors = []
            for x in self.interval_layer.test_act_buffer:
                x = x.to(device)
                features_for_anchors.append(x.detach())

            # ============================================================
            # Phase 2 — Lock Anchors (Define the "Old Safe Zone")
            # ============================================================
            if features_for_anchors:
                Z_all = torch.cat(features_for_anchors, dim=0)
                
                self.learnable_relu.anchor_next_shift(
                    z=Z_all, 
                    task_id=task_id, 
                    percentile_high=0.99,
                    percentile_low=0.01
                )
                
                self.learnable_relu.set_no_used_basis_functions(task_id)

                if task_id > 1:
                    self.learnable_relu.freeze_basis_function(task_id-2)

            # ============================================================
            # Phase 3 — Reset Interval Bounds
            # ============================================================
            self.interval_layer.reset_range()
                
            log.info(f"Task {task_id} setup complete. Hinge Anchors locked for Stability.")

    def forward(self, x: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        """
        Augment task loss with Stability Regularization.

        Components:
        1. Variance Loss: Keeps feature representations compact.
        2. Feature Drift: Keeps backbone outputs similar to old prompt outputs.
        3. Hinge Drift: Pins the Linear Head at the boundary of old tasks.
        """

        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        var_loss = zero.clone()
        drift_loss = zero.clone()
        feature_drift_loss = zero.clone()

        acts = self.interval_layer.curr_task_last_batch
        acts_flat = acts.view(acts.size(0), -1)
        batch_var = acts_flat.var(dim=0, unbiased=False).mean()
        var_loss += batch_var

        if self.task_id > 0:
            with torch.no_grad():
                q, _ = self.feature_extractor(x)
                q = q[:,0,:]
            
            y_old, _ = self.feature_extractor(
                x, prompt=self.old_prompt, q=q, train=False, task_id=self.task_id
            )
            y_old = y_old[:,0,:].detach()

            lb = self.interval_layer.min.to(x.device)
            ub = self.interval_layer.max.to(x.device)
            mask = ((acts >= lb) & (acts <= ub)).float()
            
            feature_drift_loss += (
                (mask * (y_old - acts).pow(2)).sum() / (mask.sum() + 1e-8)
            )
            
            delta_W = self.curr_linear_layer.weight - self.prev_linear_layer.weight
            delta_b = self.curr_linear_layer.bias - self.prev_linear_layer.bias
            
            dW_pos = torch.relu(delta_W)
            dW_neg = torch.relu(-delta_W)

            if self.task_id == 1:
                lb_trans = self.learnable_relu(lb.unsqueeze(0)).squeeze(0)
                ub_trans = self.learnable_relu(ub.unsqueeze(0)).squeeze(0)
                
                d_l = (dW_pos @ lb_trans) - (dW_neg @ ub_trans) + delta_b
                d_u = (dW_pos @ ub_trans) - (dW_neg @ lb_trans) + delta_b
                
                drift_loss += (d_l.pow(2).mean() + d_u.pow(2).mean())

            else:
                num_old_hinges = self.learnable_relu.num_active_hinges - 1
                
                breakpoints = [lb, ub]
                
                c_r = self.learnable_relu.c_r[:num_old_hinges].squeeze(1)
                c_l = self.learnable_relu.c_l[:num_old_hinges].squeeze(1)
                
                breakpoints.extend([c_r[k] for k in range(num_old_hinges)])
                breakpoints.extend([c_l[k] for k in range(num_old_hinges)])

                breaks_stack = torch.stack(breakpoints, dim=0)
                sorted_breaks, _ = torch.sort(breaks_stack, dim=0)
                
                for j in range(sorted_breaks.size(0) - 1):
                    l_seg = sorted_breaks[j]
                    u_seg = sorted_breaks[j+1]

                    out_l = self.learnable_relu(l_seg.unsqueeze(0)).squeeze(0)
                    out_u = self.learnable_relu(u_seg.unsqueeze(0)).squeeze(0)

                    d_l = (dW_pos @ out_l) - (dW_neg @ out_u) + delta_b
                    d_u = (dW_pos @ out_u) - (dW_neg @ out_l) + delta_b
                    
                    drift_loss += (d_l.pow(2) + d_u.pow(2)).mean()

        return loss + \
                self.lambda_var * var_loss + \
                self.lambda_drift * drift_loss + \
                self.lambda_feat * feature_drift_loss