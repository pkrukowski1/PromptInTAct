from copy import deepcopy, copy
from typing import Union
from collections import OrderedDict

import torch
import torch.nn as nn

from models.layers.interval_activation import IntervalActivation
from models.zoo import L2P, DualPrompt, CodaPrompt

class IntervalPenalization(nn.Module):

    def __init__(self,
            var_scale: float = 0.01,
            output_reg_scale: float = 1.0,
            interval_drift_reg_scale: float = 1.0
        ) -> None:
        
        super().__init__()
        self.task_id = None

        self.var_scale = var_scale
        self.output_reg_scale = output_reg_scale
        self.interval_drift_reg_scale = interval_drift_reg_scale

        self.params_buffer = {}

        self.curr_classifier_head = None
        self.old_classifier_head = None
        self.feature_extractor = None

        self.prompt = None

    def detach_interval_last_batches(self, curr_classifier_head):
        layers = list(curr_classifier_head.children())
        for layer in layers:
            if isinstance(layer, IntervalActivation):
                if layer.curr_task_last_batch is not None:
                    layer.curr_task_last_batch = []


    def setup_task(self, task_id: int, curr_classifier_head: nn.Sequential, 
                   feature_extractor: nn.Sequential, prompt: Union[CodaPrompt,L2P,DualPrompt]) -> None:

        self.task_id = task_id
        self.curr_classifier_head = curr_classifier_head
        self.prompt = prompt

        if task_id > 0:
            self.params_buffer = {
                name: p.detach().clone()
                for name, p in self.curr_classifier_head.named_parameters()
        }

            self.detach_interval_last_batches(curr_classifier_head)
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

        layers = list(self.curr_classifier_head.children())
        interval_act_layers = [i for i, layer in enumerate(layers) if isinstance(layer, IntervalActivation)]

        var_loss = torch.tensor(0.0, dtype=float).to(x.device)
        output_reg_loss = torch.tensor(0.0, dtype=float).to(x.device)
        interval_drift_loss = torch.tensor(0.0, dtype=float).to(x.device)

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

                lower_bound_reg = torch.tensor(0.0, dtype=float).to(x.device)
                upper_bound_reg = torch.tensor(0.0, dtype=float).to(x.device)
                layer_prefix = f"{idx+1}"  # assuming [Interval, Linear, Interval, Linear, ...] ordering

                for name, p in next_layer.named_parameters():
                    full_name = f"{layer_prefix}.{name}"
                    if full_name in self.params_buffer:
                        prev_param = self.params_buffer[full_name]

                        if "weight" in name:
                            weight_diff = p - prev_param

                            weight_diff_pos = torch.relu(weight_diff)
                            weight_diff_neg = torch.relu(-weight_diff)

                            # Reduce to scalar so it matches lower_bound_reg
                            lower_bound_reg += (lb @ weight_diff_pos.T - ub @ weight_diff_neg.T).sum()
                            upper_bound_reg += (ub @ weight_diff_pos.T - lb @ weight_diff_neg.T).sum()

                        elif "bias" in name:
                            diff = p - prev_param
                            lower_bound_reg += diff.sum()
                            upper_bound_reg += diff.sum()

                        output_reg_loss += lower_bound_reg.sum().pow(2) + upper_bound_reg.sum().pow(2)
        loss = (
            loss
            + self.var_scale * var_loss
            + self.output_reg_scale * output_reg_loss
            + self.interval_drift_reg_scale * interval_drift_loss
        )
        return loss
