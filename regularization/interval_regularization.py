from copy import deepcopy
from typing import Union

import torch
import torch.nn as nn

from models.layers.interval_activation import IntervalActivation
from models.zoo import L2P, DualPrompt, CodaPrompt

class IntervalPenalization(nn.Module):
    """
    Continual learning regularizer that protects representations learned inside 
    `IntervalActivation` hypercubes across tasks.

    This class adds multiple penalties to the task loss:
    
    - **Variance loss (`var_scale`)**  
      Minimizes activation variance inside each interval, encouraging stable 
      and compact representations.
    
    - **Output preservation loss (`output_reg_scale`)**  
      Constrains parameters above an `IntervalActivation` to keep producing 
      similar outputs for previously learned intervals.
    
    - **Interval drift loss (`interval_drift_reg_scale`)**  
      Penalizes deviations of new activations from old-task activations 
      inside the same hypercube, with a stronger penalty near the cube center.

    Together, these terms reduce representation drift inside protected regions, 
    while still allowing free adaptation outside.

    Attributes:
        var_scale (float): Weight of the variance regularizer.
        output_reg_scale (float): Weight of the output preservation term.
        interval_drift_reg_scale (float): Weight of the drift regularizer.
        task_id (int): Identifier of the current task.
        params_buffer (dict): Snapshot of frozen parameters from the previous task.
        old_state (dict): Full parameter/buffer snapshot used for drift comparison.

    Methods:
        setup_task(task_id):
            Prepares state before starting a new task (snapshots old params/buffers).
        forward_with_snapshot(x, stop_at="IntervalActivation"):
            Runs a forward pass with frozen params up to the first IntervalActivation.
        snapshot_state():
            Creates a snapshot of all parameters and buffers.
        forward(x, y, loss, preds):
            Adds interval regularization terms to the given loss.
    """

    def __init__(self,
            var_scale: float = 0.01,
            output_reg_scale: float = 1.0,
            interval_drift_reg_scale: float = 1.0
        ) -> None:
        """
        Initialize the interval penalization plugin.

        Args:
            var_scale (float, optional): Weight of the variance penalty. Default: 0.01.
            output_reg_scale (float, optional): Weight of the output preservation penalty. Default: 1.0.
            interval_drift_reg_scale (float, optional): Weight of the interval drift penalty. Default: 1.0.
        """
        
        super().__init__()
        self.task_id = None

        self.var_scale = var_scale
        self.output_reg_scale = output_reg_scale
        self.interval_drift_reg_scale = interval_drift_reg_scale

        self.input_shape = None
        self.params_buffer = {}

        self.curr_classifier_head = None
        self.old_classifier_head = None
        self.old_feature_extractor = None

        self.prompt = None


    def setup_task(self, task_id: int, curr_classifier_head: nn.Sequential, 
                   curr_feature_extractor: nn.Sequential, prompt: Union[CodaPrompt,L2P,DualPrompt]) -> None:
        """
        Prepare the plugin for a new task.

        - On task 0: only sets task id.
        - On later tasks: freezes parameters, saves previous params to `params_buffer`,
        and snapshots full state into `old_classifier_head`.
        """
        self.task_id = task_id
        self.curr_classifier_head = curr_classifier_head
        self.prompt = prompt

        if task_id > 0:
            self.params_buffer = {}
            for name, p in self.curr_classifier_head.named_parameters():
                if p.requires_grad:
                    self.params_buffer[name] = p.detach().clone()
                    p.requires_grad = False

            self.old_classifier_head = deepcopy(self.curr_classifier_head)
            for p in self.old_classifier_head.parameters():
                p.requires_grad = False

            self.old_feature_extractor = deepcopy(curr_feature_extractor)
            for p in self.old_feature_extractor.parameters():
                p.requires_grad = False

            for layer in self.curr_classifier_head:
                if isinstance(layer, IntervalActivation):
                    layer.reset_range()

                    
    def forward(self, x: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        """
        Add interval regularization penalties to the current loss.  

        Penalties:
            - Variance loss: discourages variance within interval activations.  
            - Drift loss: penalizes change of activations inside the old-task hypercube.  
            - Output reg: discourages parameter updates that break interval consistency.  

        Args:
            x (torch.Tensor): Input tensor.  
            loss (torch.Tensor): Current task loss.  

        Returns:
            (loss, preds): Updated loss with added penalties, predictions unchanged.
        """

        x = x.flatten(start_dim=1)
        self.input_shape = x.shape

        layers = self.curr_classifier_head.modules()
        interval_act_layers = [layer for layer in layers if isinstance(layer, IntervalActivation)]

        var_loss = torch.tensor(0.0, dtype=float).to(x.device)
        output_reg_loss = torch.tensor(0.0, dtype=float).to(x.device)
        interval_drift_loss = torch.tensor(0.0, dtype=float).to(x.device)

        for idx, layer in enumerate(interval_act_layers):
            acts = layer.curr_task_last_batch
            acts_flat = acts.view(acts.size(0), -1)
            batch_var = acts_flat.var(dim=0, unbiased=False).mean()
            var_loss += batch_var

            if self.task_id > 0:
                lb = layer.min.to(x.device)
                ub = layer.max.to(x.device)

                # Drift only at the FIRST IntervalActivation
                if idx == 0:
                    with torch.no_grad():
                        q, _ = self.old_feature_extractor(x)
                        q = q[:,0,:]
                    y_old, _ = self.old_feature_extractor(x, prompt=self.prompt, q=q, train=False, task_id=self.task_id)
                    y_old = y_old[:,0,:]

                    mask = ((acts >= lb) & (acts <= ub)).float()
                    interval_drift_loss += (
                        (mask * (y_old - acts).pow(2)).sum() / (mask.sum() + 1e-8)
                    )

                # Output reg at this interval (first and all above)
                # In pattern [Interval, Linear, Interval, Linear, ...],
                # the *next* Linear belongs to this Interval
                next_layer = layers[2*idx+1]  

                lower_bound_reg = torch.tensor(0.0, dtype=float).to(x.device)
                upper_bound_reg = torch.tensor(0.0, dtype=float).to(x.device)
                for name, p in next_layer.named_parameters():
                    for mod_name, mod_param in self.curr_classifier_head.named_parameters():
                        if mod_param is p and mod_name in self.params_buffer:
                            prev_param = self.params_buffer[mod_name]
                            if "weight" in name:
                                weight_diff = p - prev_param

                                weight_diff_pos = torch.relu(weight_diff)
                                weight_diff_neg = torch.relu(-weight_diff)

                                lower_bound_reg += weight_diff_pos @ lb - weight_diff_neg @ ub
                                upper_bound_reg += weight_diff_pos @ ub - weight_diff_neg @ lb

                            elif "bias" in name:
                                lower_bound_reg += p - prev_param
                                upper_bound_reg += p - prev_param

                output_reg_loss += lower_bound_reg.sum().pow(2) + upper_bound_reg.sum().pow(2)

        loss = (
            loss
            + self.var_scale * var_loss
            + self.output_reg_scale * output_reg_loss
            + self.interval_drift_reg_scale * interval_drift_loss
        )
        return loss
