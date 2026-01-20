from copy import deepcopy

import torch
import torch.nn as nn

from models.layers.interval_activation import IntervalActivation
from .utils import detach_interval_last_batches

class InTActRegularizationWithoutFE(nn.Module):
    """
    InTAct regularization loss module for continual learning.

    Adds interval-based penalties to the task loss to:
        - Reduce activation variance within interval layers
        - Penalize classifier weight drift across tasks
        - Limit interval-induced output drift
        - Optionally align new activation centers with previous bounds
    """

    def __init__(self,
            lambda_var: float = 0.01,
            lambda_drift: float = 1.0,
            use_align_loss: bool = True
        ) -> None:
        """
        Initialize the InTAct regularization module.

        Args:
            lambda_var (float): Weight for activation variance loss.
            lambda_drift (float): Weight for weight/output drift loss.
            use_align_loss (bool): Enable activation center alignment loss.
        """
        
        super().__init__()
        self.task_id = None

        self.lambda_var = lambda_var
        self.lambda_drift = lambda_drift
        self.use_align_loss = use_align_loss

        self.params_buffer = {}

        self.curr_classifier_head = None
        self.old_classifier_head = None

    def setup_task(
        self,
        task_id: int,
        curr_classifier_head: nn.Sequential,
    ) -> None:
        """
        Prepare regularization state for a new task.

        For tasks > 0, this:
            - Stores a frozen copy of previous classifier parameters
            - Detaches interval activation buffers
            - Resets interval ranges for the new task

        Args:
            task_id (int): Current task index.
            curr_classifier_head (nn.Sequential): Classifier head for the task.
        """

        self.task_id = task_id
        self.curr_classifier_head = curr_classifier_head

        if task_id > 0:
            self.params_buffer = {
                name: p.detach().clone()
                for name, p in self.curr_classifier_head.named_parameters()
        }

            detach_interval_last_batches(curr_classifier_head)
            self.old_classifier_head = deepcopy(curr_classifier_head)
            for p in self.old_classifier_head.parameters():
                p.requires_grad = False


            for layer in self.curr_classifier_head:
                if isinstance(layer, IntervalActivation):
                    layer.reset_range()
                    
    def forward(self, x: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        """
        Compute InTAct regularization and add it to the task loss.

        Args:
            x (torch.Tensor): Input batch tensor.
            loss (torch.Tensor): Task-specific loss.

        Returns:
            torch.Tensor: Loss augmented with InTAct penalties.
        """

        layers = list(self.curr_classifier_head.children())
        interval_act_layers = [i for i, layer in enumerate(layers) if isinstance(layer, IntervalActivation)]

        zero = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        var_loss = zero.clone()
        output_reg_loss = zero.clone()
        align_repr_loss = zero.clone()


        for idx in interval_act_layers:
            acts = layers[idx].curr_task_last_batch

            acts_flat = acts.view(acts.size(0), -1)
            batch_var = acts_flat.var(dim=0, unbiased=False).mean()
            var_loss += batch_var

            if self.task_id > 0:
                lb = layers[idx].min.to(x.device)
                ub = layers[idx].max.to(x.device)

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
            + self.lambda_drift * output_reg_loss
            + align_repr_loss
        )
        return loss
