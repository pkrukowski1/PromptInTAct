from copy import deepcopy
from typing import Union

import torch
import torch.nn as nn

from models.layers.interval_activation import IntervalActivation
from models.zoo import L2P, DualPrompt, CodaPrompt
from .utils import detach_interval_last_batches

class InTActRegularization(nn.Module):
    """
    Loss module for usage of InTAct.

    This module penalizes:
        - Variance within interval activations of the current task.
        - Drift in internal representations between tasks.
        - Feature-level drift.
        - Misalignment of new representations relative to previous interval bounds.

    Attributes:
        task_id (int | None): Current task index.
        lambda_var (float): Scale factor for variance regularization loss.
        lambda_int_drift (float): Scale factor for output / weight drift loss.
        lambda_feat (float): Scale factor for internal representation drift loss.
        use_align_loss (bool): Whether to apply center-alignment loss for activations.
        params_buffer (dict): Stores cloned parameters from previous task for regularization.
        curr_classifier_head (nn.Sequential | None): Current task classifier head.
        old_classifier_head (nn.Sequential | None): Classifier head from previous task (frozen).
        feature_extractor (nn.Sequential | None): Shared feature extractor.
        prompt (Union[CodaPrompt, L2P, DualPrompt] | None): Current prompt module.
        old_prompt (Union[CodaPrompt, L2P, DualPrompt] | None): Previous task prompt module (frozen).
    """

    def __init__(self,
            lambda_var: float = 0.01,
            lambda_int_drift: float = 1.0,
            lambda_feat: float = 1.0,
            use_align_loss: bool = True
        ) -> None:
        """
        Initializes InTActRegularization with specified loss scales.

        Args:
            lambda_var (float, optional): Scale factor for variance loss. Defaults to 0.01.
            lambda_int_drift (float, optional): Scale factor for output / weight drift loss. Defaults to 1.0.
            lambda_feat (float, optional): Scale factor for feature drift loss. Defaults to 1.0.
            use_align_loss (bool, optional): Whether to include activation center alignment loss. Defaults to True.
        """
        
        super().__init__()
        self.task_id = None

        self.lambda_var = lambda_var
        self.lambda_int_drift = lambda_int_drift
        self.lambda_feat = lambda_feat
        self.use_align_loss = use_align_loss

        self.params_buffer = {}

        self.curr_classifier_head = None
        self.old_classifier_head = None
        self.feature_extractor = None

        self.prompt = None


    def setup_task(
        self,
        task_id: int,
        curr_classifier_head: nn.Sequential,
        feature_extractor: nn.Sequential,
        prompt: Union[CodaPrompt, L2P, DualPrompt]
    ) -> None:
        """
        Sets up the penalization module for the current task.

        Clones previous task parameters and prompts, freezes them, and resets
        interval activation bounds for the current classifier head.

        Args:
            task_id (int): Index of the current task.
            curr_classifier_head (nn.Sequential): Classifier head for current task.
            feature_extractor (nn.Sequential): Shared feature extractor (frozen).
            prompt (CodaPrompt | L2P | DualPrompt): Prompt module for the current task.
        """

        self.task_id = task_id
        self.curr_classifier_head = curr_classifier_head
        self.prompt = prompt

        if task_id > 0:
            self.params_buffer = {
                name: p.detach().clone()
                for name, p in self.curr_classifier_head.named_parameters()
        }

            detach_interval_last_batches(curr_classifier_head)
            self.old_classifier_head = deepcopy(curr_classifier_head)
            for p in self.old_classifier_head.parameters():
                p.requires_grad = False

            # Feature extractor is shared and frozen, so we just keep a reference to it
            self.feature_extractor = feature_extractor

            self.old_prompt = deepcopy(self.prompt)
            for p in self.old_prompt.parameters():
                p.requires_grad = False

            for layer in self.curr_classifier_head:
                if isinstance(layer, IntervalActivation):
                    layer.reset_range()
                    
    def forward(self, x: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        """
        Computes the interval-based penalization loss and adds it to the task loss.

        Loss components:
            - Variance loss: Encourages consistent activations within a batch.
            - Output regularization: Penalizes changes in weights and biases relative to previous task.
            - Interval drift loss: Penalizes deviations of current activations from previous task outputs.
            - Alignment loss: Penalizes shift in activation centers relative to previous interval bounds.

        Args:
            x (torch.Tensor): Input batch tensor.
            loss (torch.Tensor): Original task-specific loss to augment.

        Returns:
            torch.Tensor: Loss augmented with interval penalization terms.
        """

        layers = list(self.curr_classifier_head.children())
        interval_act_layers = [i for i, layer in enumerate(layers) if isinstance(layer, IntervalActivation)]

        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        var_loss = zero.clone()
        output_reg_loss = zero.clone()
        interval_drift_loss = zero.clone()
        align_repr_loss = zero.clone()


        for idx in interval_act_layers:
            acts = layers[idx].curr_task_last_batch

            acts_flat = acts.view(acts.size(0), -1)
            batch_var = acts_flat.var(dim=0, unbiased=False).mean()
            var_loss += batch_var

            if self.task_id > 0:
                lb = layers[idx].min.to(x.device)
                ub = layers[idx].max.to(x.device)

                # Drift only at the FIRST IntervalActivation
                if idx == interval_act_layers[0]:
                    with torch.no_grad():
                        q, _ = self.feature_extractor(x)
                        q = q[:,0,:]
                    y_old, _ = self.feature_extractor(x, prompt=self.old_prompt, q=q, train=False, task_id=self.task_id)
                    y_old = y_old[:,0,:].detach()

                    mask = ((acts >= lb) & (acts <= ub)).float()
                    interval_drift_loss += (
                        (mask * (y_old - acts).pow(2)).sum() / (mask.sum() + 1e-8)
                    )

                if idx + 1 < len(layers) and isinstance(layers[idx+1], torch.nn.Linear):
                    next_layer = layers[idx+1]

                layer_prefix = f"{idx+1}"  # assuming [Interval, Linear, Interval, Linear, ...] ordering

                out_dim = next(next_layer.parameters()).shape[0]
                total_lower = torch.zeros(out_dim, device=x.device)
                total_upper = torch.zeros(out_dim, device=x.device)

                for name, p in next_layer.named_parameters():
                    full_name = f"{layer_prefix}.{name}"
                    if full_name in self.params_buffer:
                        prev_param = self.params_buffer[full_name]
                        diff = p - prev_param

                        if "weight" in name:
                            diff_pos = torch.relu(diff)
                            diff_neg = torch.relu(-diff)

                            total_lower += (diff_pos @ lb - diff_neg @ ub)
                            total_upper += (diff_pos @ ub - diff_neg @ lb)

                        elif "bias" in name:
                            total_lower += diff
                            total_upper += diff

                output_reg_loss += (total_lower.pow(2).mean() + total_upper.pow(2).mean())

                if self.use_align_loss:
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
        loss = (
            loss
            + self.lambda_var * var_loss
            + self.lambda_int_drift * output_reg_loss
            + self.lambda_feat * interval_drift_loss
            + align_repr_loss
        )
        return loss
