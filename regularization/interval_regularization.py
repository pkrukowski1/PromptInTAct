import torch
import torch.nn as nn
import numpy as np
import itertools

from models.layers.interval_activation import IntervalActivation
from models.zoo import L2P, DualPrompt, CodaPrompt

from copy import deepcopy
from typing import Union

class IntervalPenalization(nn.Module):
    """
    Loss module for usage of InTAct.

    This module penalizes:
        - Variance within interval activations of the current task.
        - Drift in internal representations between tasks.
        - Feature-level drift.
        - Misalignment of new representations relative to previous interval bounds.

    Attributes:
        task_id (int | None): Current task index.
        var_loss_scale (float): Scale factor for variance regularization loss.
        internal_repr_drift_loss_scale (float): Scale factor for output / weight drift loss.
        feature_loss_scale (float): Scale factor for internal representation drift loss.
        use_align_loss (bool): Whether to apply center-alignment loss for activations.
        params_buffer (dict): Stores cloned parameters from previous task for regularization.
        curr_classifier_head (nn.Sequential | None): Current task classifier head.
        old_classifier_head (nn.Sequential | None): Classifier head from previous task (frozen).
        feature_extractor (nn.Sequential | None): Shared feature extractor.
        prompt (Union[CodaPrompt, L2P, DualPrompt] | None): Current prompt module.
        old_prompt (Union[CodaPrompt, L2P, DualPrompt] | None): Previous task prompt module (frozen).
    """

    def __init__(self,
            var_loss_scale: float = 0.01,
            internal_repr_drift_loss_scale: float = 1.0,
            feature_loss_scale: float = 1.0,
            use_align_loss: bool = True,
            use_metrics: bool = False,
            gradient_tracker=None,
        ) -> None:
        """
        Initializes IntervalPenalization with specified loss scales.

        Args:
            var_loss_scale (float, optional): Scale factor for variance loss. Defaults to 0.01.
            internal_repr_drift_loss_scale (float, optional): Scale factor for output / weight drift loss. Defaults to 1.0.
            feature_loss_scale (float, optional): Scale factor for feature drift loss. Defaults to 1.0.
            use_align_loss (bool, optional): Whether to include activation center alignment loss. Defaults to True.
            use_metrics (bool, optional): If True, track occupancy ratio and constraint activation rate. Defaults to False.
            gradient_tracker: GradientCosineTracker or None.  When set, per-component cosine sims are logged each step.
        """

        super().__init__()
        self.task_id = None

        self.var_loss_scale = var_loss_scale
        self.internal_repr_drift_loss_scale = internal_repr_drift_loss_scale
        self.feature_loss_scale = feature_loss_scale
        self.use_align_loss = use_align_loss
        self.use_metrics = use_metrics
        self.gradient_tracker = gradient_tracker

        self.params_buffer = {}

        self.curr_classifier_head = None
        self.old_classifier_head = None
        self.feature_extractor = None

        self.prompt = None

        self._task_box_history = []
        self._task_cur_min = []
        self._task_cur_max = []
        self._violation_sum = 0.0
        self._sample_sum = 0.0
        self._n_interval_layers = 0

        self.cumulative_box_history = []

    def detach_interval_last_batches(self, curr_classifier_head: nn.Sequential) -> None:
        """
        Clears the stored last batch activations in all IntervalActivation layers
        of the current classifier head.

        Args:
            curr_classifier_head (nn.Sequential): Classifier head containing IntervalActivation layers.
        """
        layers = list(curr_classifier_head.children())
        for layer in layers:
            if isinstance(layer, IntervalActivation):
                if layer.curr_task_last_batch is not None:
                    layer.curr_task_last_batch = []


    def setup_task(
        self,
        task_id: int,
        curr_classifier_head: nn.Sequential,
        feature_extractor: nn.Sequential,
        prompt: Union[CodaPrompt, L2P, DualPrompt]
    ) -> None:
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

            self.feature_extractor = feature_extractor

            self.old_prompt = deepcopy(self.prompt)
            for p in self.old_prompt.parameters():
                p.requires_grad = False

            for idx, layer in enumerate(self.curr_classifier_head):
                if isinstance(layer, IntervalActivation):
                    self._snapshot_layer(task_id - 1, idx, layer)
                    layer.reset_range()
                    self._snapshot_cumulative_bounds(task_id - 1, idx, layer)
                    print(f"Volume of the cumulative hypercube for {idx+1}-th layer in classification head: {torch.mean(layer.max - layer.min).item()}")

        layers = list(self.curr_classifier_head.children())
        self._n_interval_layers = sum(1 for l in layers if isinstance(l, IntervalActivation))
        self._task_cur_min = [None] * self._n_interval_layers
        self._task_cur_max = [None] * self._n_interval_layers
        self._violation_sum = 0.0
        self._sample_sum = 0.0

    def _snapshot_layer(self, task_id: int, layer_idx: int, layer: IntervalActivation) -> None:
        if not self.use_metrics or task_id < 0:
            return
        tmin, tmax = self._get_task_bounds(layer)
        while len(self._task_box_history) <= task_id:
            self._task_box_history.append([])
        boxes = self._task_box_history[task_id]
        while len(boxes) <= layer_idx:
            boxes.append(None)
        boxes[layer_idx] = (tmin, tmax)

    def _snapshot_cumulative_bounds(self, task_id: int, layer_idx: int, layer: IntervalActivation) -> None:
        if not self.use_metrics:
            return
        while len(self.cumulative_box_history) <= task_id:
            self.cumulative_box_history.append([])
        boxes = self.cumulative_box_history[task_id]
        while len(boxes) <= layer_idx:
            boxes.append(None)
        if layer.min is not None and layer.max is not None:
            boxes[layer_idx] = (layer.min.clone(), layer.max.clone())

    def _current_cumulative_bounds(self):
        layers = list(self.curr_classifier_head.children())
        result = []
        for layer in layers:
            if isinstance(layer, IntervalActivation):
                tmin, tmax = self._get_task_bounds(layer)
                if tmin is not None and tmax is not None:
                    if layer.min is not None and layer.max is not None:
                        tmin = torch.minimum(layer.min.cpu(), tmin)
                        tmax = torch.maximum(layer.max.cpu(), tmax)
                elif layer.min is not None and layer.max is not None:
                    tmin, tmax = layer.min.cpu().clone(), layer.max.cpu().clone()
                result.append((tmin, tmax))
        return result

    def compute_metrics(self) -> dict:
        result = {}
        if not self.use_metrics:
            return result
        layers = list(self.curr_classifier_head.children())
        cum_boxes_info = []
        for idx, l in enumerate(layers):
            if isinstance(l, IntervalActivation):
                if l.min is not None and l.max is not None:
                    cum_boxes_info.append((idx, l.min.clone(), l.max.clone()))
        if not cum_boxes_info:
            return result
        eps = 1e-10
        cum_log_vols = []
        cum_sides_cpu = []
        for _, cm, cM in cum_boxes_info:
            sides = (cM - cm).clamp(min=eps).cpu()
            cum_log_vols.append(torch.sum(torch.log(sides)).item())
            cum_sides_cpu.append(sides)
        for layer_i, (seq_idx, _, _) in enumerate(cum_boxes_info):
            cur_log_vol = cum_log_vols[layer_i]
            cur_sides = cum_sides_cpu[layer_i]
            task_boxes_for_layer = []
            for task_idx, task_boxes in enumerate(self._task_box_history):
                if len(task_boxes) <= seq_idx:
                    continue
                box = task_boxes[seq_idx]
                if box is None or box[0] is None:
                    continue
                tmin, tmax = box
                log_V_i = torch.sum(torch.log((tmax - tmin).clamp(min=eps))).item()
                log_ratio_i = log_V_i - cur_log_vol
                V_ratio_i = float(np.exp(log_ratio_i))
                result[f"V_ratio_task{task_idx}_layer{layer_i}"] = V_ratio_i
                result[f"log_ratio_task{task_idx}_layer{layer_i}"] = log_ratio_i
                task_boxes_for_layer.append((tmin, tmax))
            n_tasks = len(task_boxes_for_layer)
            if n_tasks > 0:
                cum_sides = cur_sides.double()
                V_ratio_total = torch.tensor(0.0, dtype=torch.float64)
                for k in range(1, n_tasks + 1):
                    sign = 1.0 if k % 2 == 1 else -1.0
                    for subset in itertools.combinations(range(n_tasks), k):
                        mins = torch.stack([task_boxes_for_layer[i][0].double() for i in subset])
                        maxs = torch.stack([task_boxes_for_layer[i][1].double() for i in subset])
                        lower = mins.max(dim=0).values
                        upper = maxs.min(dim=0).values
                        overlap = (upper - lower).clamp(min=0.0)
                        if (overlap == 0.0).any():
                            continue
                        log_ratio_sum = torch.sum(
                            torch.log((overlap / cum_sides).clamp(min=eps))
                        ).item()
                        if log_ratio_sum > -745:
                            V_ratio_total += sign * float(np.exp(log_ratio_sum))
                V_ratio = max(0.0, V_ratio_total.item())
                if V_ratio > 0:
                    V_log_ratio = float(np.log(V_ratio))
                else:
                    V_log_ratio = float('-inf')
                result[f"V_ratio_sum_layer{layer_i}"] = V_ratio
                result[f"V_log_ratio_sum_layer{layer_i}"] = V_log_ratio
        c_rate = self._violation_sum / max(self._sample_sum, 1)
        result["C_rate"] = c_rate
        return result

    def finalize_metrics(self) -> dict:
        """Snapshot the last task and return metrics. Call after training loop completes."""
        if not self.use_metrics:
            return {}
        layers = list(self.curr_classifier_head.children())
        for idx, layer in enumerate(layers):
            if isinstance(layer, IntervalActivation):
                self._snapshot_layer(self.task_id, idx, layer)
                self._snapshot_cumulative_bounds(self.task_id, idx, layer)
        return self.compute_metrics()

    def _get_task_bounds(self, layer: IntervalActivation):
        """Compute per-task (min, max) from test_act_buffer, always on CPU."""
        if len(layer.test_act_buffer) == 0:
            return None, None
        activations = torch.cat(layer.test_act_buffer, dim=0)
        n = activations.size(0)
        if n == 0:
            return None, None
        sorted_buf, _ = torch.sort(activations, dim=0)
        l_idx = int(np.clip(int(n * layer.lower_percentile), 0, n - 1))
        u_idx = int(np.clip(int(n * layer.upper_percentile), 0, n - 1))
        return sorted_buf[l_idx].clone(), sorted_buf[u_idx].clone()

    def _track_metrics(self, acts: torch.Tensor, layer_idx: int, lb, ub) -> None:
        if not self.use_metrics:
            return
        with torch.no_grad():
            bmin = acts.min(dim=0).values.cpu()
            bmax = acts.max(dim=0).values.cpu()
            if self._task_cur_min[layer_idx] is None:
                self._task_cur_min[layer_idx] = bmin
                self._task_cur_max[layer_idx] = bmax
            else:
                self._task_cur_min[layer_idx] = torch.minimum(self._task_cur_min[layer_idx], bmin)
                self._task_cur_max[layer_idx] = torch.maximum(self._task_cur_max[layer_idx], bmax)
            if lb is not None and ub is not None:
                violated = ((acts < lb) | (acts > ub)).sum().item()
                self._violation_sum += violated
                self._sample_sum += acts.numel()
        

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


        for layer_i, idx in enumerate(interval_act_layers):
            acts = layers[idx].curr_task_last_batch

            acts_flat = acts.view(acts.size(0), -1)
            batch_var = acts_flat.var(dim=0, unbiased=False).mean()
            var_loss += batch_var

            if self.task_id > 0:
                lb = layers[idx].min.to(x.device)
                ub = layers[idx].max.to(x.device)
                self._track_metrics(acts_flat, layer_i, lb, ub)

                # Drift only at the FIRST IntervalActivation
                if idx == interval_act_layers[0]:
                    with torch.no_grad():
                        q, _ = self.feature_extractor(x)
                        q = q[:,0,:]
                    y_old_raw, _ = self.feature_extractor(x, prompt=self.old_prompt, q=q, train=False, task_id=self.task_id)
                    y_old_raw = y_old_raw[:,0,:].detach()
                    y_old = layers[idx - 1](y_old_raw)

                    mask = ((acts >= lb) & (acts <= ub)).float()
                    interval_drift_loss = interval_drift_loss + (
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

                    new_lb, _ = acts_flat.min(dim=0)
                    new_ub, _ = acts_flat.max(dim=0)
                    new_center = (new_ub + new_lb) / 2.0

                    center_loss = torch.norm(new_center - prev_center, p=2)

                    align_repr_loss = align_repr_loss + center_loss / (prev_radii.mean() + 1e-8)
        scaled_var = self.var_loss_scale * var_loss
        scaled_output_reg = self.internal_repr_drift_loss_scale * output_reg_loss
        scaled_interval_drift = self.feature_loss_scale * interval_drift_loss

        total = (
            loss
            + scaled_var
            + scaled_output_reg
            + scaled_interval_drift
            + align_repr_loss
        )

        if self.gradient_tracker is not None:
            loss_dict: dict[str, torch.Tensor] = {
                "ce": loss,
                "var": scaled_var,
                "output_reg": scaled_output_reg,
                "interval_drift": scaled_interval_drift,
                "align": align_repr_loss,
            }
            self.gradient_tracker.record(loss_dict)

        return total
