import logging
from typing import Tuple, Union
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.learnable_relu import LearnableReLU
from models.layers.interval_activation import IntervalActivation
from models.zoo import L2P, DualPrompt, CodaPrompt
from .utils import detach_interval_last_batches

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class InTActPlusPlusRegularization(nn.Module):
    """
    Continual learning regularizer that protects learned representations 
    inside activation hypercubes across tasks using Interval Arithmetic (IA).

    Mathematical Pipeline:
        1. **Subspace Projection (SVD):** Projects high-dimensional inputs into a shared 
           rotated subspace M to maintain historical context across task transitions.
        2. **Interval Bounding:** Establishes robust hypercubes [min, max] around 
           activations, unioned across tasks to prevent representational drift.
        3. **Drift Regularization:** Penalizes the Mean Squared Error (MSE) of interval 
           endpoints for pre-activations, ensuring stability of learned mappings.
        4. **Monotone Expansion:** Uses LearnableReLU basis functions to grow capacity 
           without breaking the non-decreasing derivative property (monotonicity).
        5. **Slope Regularization:** Penalizes the magnitude of new basis coefficients 
           (a_t) to prevent high-gain gradients from destabilizing prior tasks.

    Penalties:
        - **Variance Loss (`var_scale`):** Encourages compact activations to minimize 
          hypercube volume, leaving 'free space' for future tasks.
        - **Representation Drift Loss (`lambda_int_drift` / `lambda_int_input`):** Strictly preserves pre-activations for previous tasks using IA endpoint MSE.
        - **Basis Regularization (`lambda_basis`):** Controls model capacity by 
          keeping new activation slopes small, favoring reuse over expansion.
    """

    def __init__(self,
            lambda_var: float = 0.01,
            lambda_slope_reg: float = 0.01,
            lambda_int_hidden_drift: float = 1.0,
            lambda_int_input_drift: float = 1.0,
            reduced_dim: int = 50,
            dil_mode: bool = False,
            regularize_classifier: bool = False,
        ) -> None:
        """
        Initialize the interval penalization plugin for continual learning.

        Args:
            lambda_var (float, optional): Weight of variance regularization. Default: 0.01.
            lambda_int_hidden_drift (float, optional): Weight of interval drift preservation for hidden layers. Default: 1.0.
            lambda_int_input_drift (float, optional): Weight of interval drift preservation for input layer. Default: 1.0.
            reduced_dim (int, optional): Dimension of the random projection space for input hypercubes. Default: 50.
            dil_mode (bool, optional): If True, classifier head is regularized (used in DIL / CIL scenarios). Default: False.
            regularize_classifier (bool, optional): If True, applies penalties to classifier head. Default: False.
        """
        
        super().__init__()
        self.task_id = None
        log.info(f"IntervalPenalization initialized with lambda_var={lambda_var}, "
                 f"lambda_int_hidden_drift={lambda_int_hidden_drift}, "
                 f"lambda_int_input_drift={lambda_int_input_drift}, "
                 f"lambda_slope_reg={lambda_slope_reg}")

        self.lambda_var = lambda_var
        self.lambda_slope_reg = lambda_slope_reg
        self.lambda_int_hidden_drift = lambda_int_hidden_drift
        self.lambda_int_input_drift = lambda_int_input_drift
        self.reduced_dim = reduced_dim

        self.dil_mode = dil_mode
        self.regularize_classifier = regularize_classifier

        self.projection_matrix = None
        self.low_dim_inputs_min = None
        self.low_dim_inputs_max = None

        self.input_mean = None  # Global mean (D,)
        self.num_samples = 0
        self.residual_max = None

        self.curr_classifier_head = None
        self.old_classifier_head = None
        self.feature_extractor = None

        self.prompt = None

        self.params_buffer = {}


    def setup_task(
        self,
        task_id: int,
        curr_classifier_head: nn.Sequential,
        feature_extractor: nn.Sequential,
        prompt: Union[CodaPrompt, L2P, DualPrompt]
    ) -> None:
        """
        Prepare the model for a new task.

        Performs:
            1. Freezing and snapshotting previously learned parameters
            2. Collecting preactivations for LearnableReLU layers
            and activations for IntervalActivation layers
            3. Computing low-dimensional projection subspace for input hypercubes
            4. Merging subspaces with previous tasks if task_id > 0
            5. Resetting activation hypercubes for IntervalActivation layers
            6. Anchoring new LearnableReLU hinges and activating additional basis functions

        Args:
            task_id (int): Index of the current task. Zero indicates the first task.
        """

        self.task_id = task_id
        self.curr_classifier_head = curr_classifier_head
        self.prompt = prompt

        device = next(curr_classifier_head).device

        if task_id == 0:
            return
        
        curr_classifier_head.eval()

        detach_interval_last_batches(curr_classifier_head)
        self.feature_extractor = feature_extractor

        # ------------------------------------------------------------
        # Phase 0: Calculate projection matrix to lower-dimensional space
        # to get hypercubes around inputs to the first layer.
        # ------------------------------------------------------------
        cls_token_repr = torch.cat([x.flatten(start_dim=1) for x in curr_classifier_head[0].test_act_buffer], dim=0).to(device)

        with torch.no_grad():
            old_mean = self.input_mean.clone() if self.input_mean is not None else torch.zeros(cls_token_repr.size(1), device=device)
            if self.input_mean is None:
                self.input_mean = cls_token_repr.mean(0)
                self.num_samples = cls_token_repr.size(0)
            else:
                n_old, n_new = self.num_samples, cls_token_repr.size(0)
                total_samples = n_old + n_new
                new_mean = cls_token_repr.mean(0)
                updated_mean = (self.input_mean * n_old + new_mean * n_new) / total_samples
                self.input_mean = updated_mean
                self.num_samples = total_samples

            # Center with global mean
            X_centered = cls_token_repr - self.input_mean

            # 2. Extract Task Basis directly from data using SVD
            U, S, Vh = torch.linalg.svd(X_centered, full_matrices=False)
            actual_dim = min(self.reduced_dim, S.size(0))
            M_task = Vh[:actual_dim, :]  # [k, D]

            if M_task.shape[0] < self.reduced_dim:
                # Pad with random orthogonal vectors if needed
                extra = self.reduced_dim - M_task.shape[0]
                random_extra = torch.randn(extra, cls_token_repr.size(1), device=device)
                Q, _ = torch.linalg.qr(random_extra.t())
                extra_basis = Q.t()[:extra]
                M_task = torch.cat([M_task, extra_basis], dim=0)

            if self.projection_matrix is None:
                self.projection_matrix = M_task.detach()
            else:
                # 3. MERGE BASES: Union of Subspaces via SVD for top directions
                B = torch.cat([self.projection_matrix.t(), M_task.t()], dim=1)  # [D, 2k]
                U, S, Vh = torch.linalg.svd(B, full_matrices=False)
                actual_dim = min(self.reduced_dim, S.size(0))
                new_projection = U[:, :actual_dim].t().detach()  # [k, D]
                if new_projection.shape[0] < self.reduced_dim:
                    extra = self.reduced_dim - new_projection.shape[0]
                    random_extra = torch.randn(extra, cls_token_repr.size(1), device=device)
                    Q, _ = torch.linalg.qr(random_extra.t())
                    extra_basis = Q.t()[:extra]
                    new_projection = torch.cat([new_projection, extra_basis], dim=0)

                similarity = self.projection_matrix @ new_projection.t()  # [k, k]
                for j in range(self.reduced_dim):
                    i = torch.argmax(torch.abs(similarity[:, j]))
                    if similarity[i, j] < 0:
                        new_projection[j] *= -1

                # Reproject old bounds to new basis (before assigning new projection)
                if self.low_dim_inputs_min is not None:
                    R = self.projection_matrix @ new_projection.t()  # (k, k)
                
                    R_t = R.t() 
                    R_pos = torch.relu(R_t)
                    R_neg = torch.relu(-R_t)

                    # Vectorized Interval Transformation
                    # new_old_min = (R_pos @ old_min - R_neg @ old_max)
                    new_old_min = torch.mv(R_pos, self.low_dim_inputs_min) - torch.mv(R_neg, self.low_dim_inputs_max)
                    new_old_max = torch.mv(R_pos, self.low_dim_inputs_max) - torch.mv(R_neg, self.low_dim_inputs_min)

                    # Adjust for global mean shift
                    shift = (old_mean - self.input_mean) @ new_projection.t()
                    new_old_min += shift
                    new_old_max += shift

                self.projection_matrix = new_projection

            z = X_centered @ self.projection_matrix.t()   # [N, k]

            all_abs_r = []
            chunk_size = 2048 
            device = X_centered.device

            for start in range(0, X_centered.size(0), chunk_size):
                end = start + chunk_size
                Xc = X_centered[start:end]
                zc = z[start:end]

                # Actual Residual: R = X - M^T z
                # This is the "information loss" of the projection
                Rc = Xc - (zc @ self.projection_matrix)
                
                all_abs_r.append(Rc.abs().cpu())

            R_abs_all = torch.cat(all_abs_r, dim=0)
            sorted_R, _ = torch.sort(R_abs_all, dim=0)
            n_r = sorted_R.size(0)
            residual_max_task = sorted_R[int(0.95 * n_r)].to(device)

            if self.residual_max is None:
                self.residual_max = residual_max_task
            else:
                self.residual_max = torch.maximum(self.residual_max, residual_max_task)

            z_cpu = z.cpu()
            sorted_z, _ = torch.sort(z_cpu, dim=0)
            n_z = sorted_z.size(0)
            
            task_min = sorted_z[int(0.05 * n_z)].to(device)
            task_max = sorted_z[int(0.95 * n_z)].to(device)

            if self.low_dim_inputs_min is None:
                self.low_dim_inputs_min = task_min
                self.low_dim_inputs_max = task_max
            else:
                self.low_dim_inputs_min = torch.minimum(new_old_min, task_min)
                self.low_dim_inputs_max = torch.maximum(new_old_max, task_max)

        # ------------------------------------------------------------
        # Phase 1: Freeze parameters
        # ------------------------------------------------------------
        self.params_buffer = {}
        self.old_classifier_head = deepcopy(curr_classifier_head)
        for p in self.old_classifier_head.parameters():
            p.requires_grad = False

        # Feature extractor is shared and frozen, so we just keep a reference to it
        self.feature_extractor = feature_extractor

        self.old_prompt = deepcopy(self.prompt)
        for p in self.old_prompt.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------
        # Phase 2: Register hooks & collect statistics
        # ------------------------------------------------------------
        preacts = {}
        hooks = []

        for idx, layer in enumerate(self.curr_classifier_head):
            if isinstance(layer, LearnableReLU):
                layer.freeze_basis_function(task_id - 1)
                preacts[idx] = []

                def preact_hook(module, inputs, outputs, idx=idx):
                    x = inputs[0]
                    z = F.linear(x, module.weight, module.bias)
                    preacts[idx].append(z.detach())

                hooks.append(layer.register_forward_hook(preact_hook))

        # ------------------------------------------------------------
        # Phase 3: Forward pass over stored data
        # ------------------------------------------------------------
        with torch.no_grad():
            for x in cls_token_repr:
                x = x.to(device)
                _ = self.curr_classifier_head(x.unsqueeze(0))

        # ------------------------------------------------------------
        # Phase 4: Update activation hypercubes
        # ------------------------------------------------------------
        for layer in self.curr_classifier_head:
            if isinstance(layer, IntervalActivation):
                layer.reset_range()

        # ------------------------------------------------------------
        # Phase 5: Anchor LearnableReLU hinges & activate new basis
        # ------------------------------------------------------------
        for idx, layer in enumerate(self.curr_classifier_head):
            if isinstance(layer, LearnableReLU):
                z_all = torch.cat(preacts[idx], dim=0)
                layer.anchor_next_shift(
                    z=z_all,
                    task_id=task_id,
                    percentile=0.95,
                )
                layer.set_no_used_basis_functions(task_id + 1)

        # ------------------------------------------------------------
        # Phase 6: Cleanup
        # ------------------------------------------------------------
        for h in hooks:
            h.remove()

        curr_classifier_head.train()

                    
    def forward(self, x: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        """
        Augments the primary task loss with geometric stability and interval drift penalties.
        
        This method implements the core 'Drift Guard' mechanism. It uses Interval Arithmetic (IA)
        to bound the output change (drift) of each layer relative to historical task data, 
        accounting for both subspace variance and reconstruction residuals.

        Mathematical Pipeline:
            1. **Variance Minimization**: Penalizes the variance of current activations 
               to encourage compact representational clustering, maximizing 'free' 
               feature space for future tasks.
            2. **Slope Regularization**: Regularizes the learnable basis coefficients 
               of the Monotone Expansion layers to prevent high-gain instability.
            3. **Input Layer Subspace Guard (Task t > 0)**:
               - Computes the drift components in the low-dimensional projection (z) 
                 and the high-dimensional residual (r).
               - Uses the stored Alignment Matrix and Global Mean Shift to bound 
                 the drift of the first layer's pre-activations.
               - Incorporates the recursive worst-case residual bound (residual_max) 
                 to ensure 'hidden' drift in the discarded SVD dimensions is penalized.
            4. **Hidden Layer Interval Guard**:
               - Propagates historical hypercubes through current weight updates (ΔW).
               - Calculates per-neuron drift boundaries [lower, upper] using IA logic:
                 δ = (ΔW+ @ lb - ΔW- @ ub) + Δb.
               - Penalizes the Mean Squared Error (MSE) of these endpoints to strictly 
                 limit functional deviation.

        Args:
            x (torch.Tensor): Input feature batch [B, D].
            loss (torch.Tensor): Current task's empirical risk (e.g., Cross-Entropy).

        Returns:
            loss (torch.Tensor): Total loss = L_task + λ_var*L_var + λ_slope*L_slope + λ_drift*L_drift.
        """

        # TODO: Do zmiany!

        layers = self.module.layers + [self.module.head]
        interval_act_layers = [layer for layer in layers if type(layer).__name__ == "IntervalActivation"]

        var_loss = torch.tensor(0.0, device=x.device)
        int_drift_loss = torch.tensor(0.0, device=x.device)
        slope_loss = torch.tensor(0.0, device=x.device)

        for idx, layer in enumerate(interval_act_layers):
            
            # Variance regularization
            acts = layer.curr_task_last_batch
            acts_flat = acts.view(acts.size(0), -1)
            batch_var = acts_flat.var(dim=0, unbiased=False).mean()
            var_loss += batch_var

            # Slope regularization
            slope = layers[2*idx].raw_scales[self.task_id]
            slope_loss += slope.pow(2).mean()

            if self.task_id > 0:
                
                lb = layer.min.to(x.device)
                ub = layer.max.to(x.device)

                if idx == 0:
                    curr_W = self.module.layers[0].weight        # [out, D]
                    curr_b = self.module.layers[0].bias          # [out]

                    prev_W, prev_b = None, None
                    for name, p in self.module.named_parameters():
                        if p is curr_W and name in self.params_buffer:
                            prev_W = self.params_buffer[name]
                        elif p is curr_b and name in self.params_buffer:
                            prev_b = self.params_buffer[name]

                    if prev_W is None or prev_b is None:
                        raise ValueError("Previous parameters for first layer not found")

                    delta_W = curr_W - prev_W                     # [out, D]
                    delta_b = curr_b - prev_b                     # [out]

                    delta_A = delta_W @ self.projection_matrix.t()   # [out, k]

                    delta_W_proj = delta_A @ self.projection_matrix  # [out, D]
                    delta_W_res = delta_W - delta_W_proj             # [out, D]

                    r_max = self.residual_max.to(x.device)           # [D]
                    res_drift_radius = delta_W_res.abs() @ r_max     # [out]

                    delta_mean = delta_W @ self.input_mean.to(x.device)

                    z_lb = self.low_dim_inputs_min.to(x.device)      # [k]
                    z_ub = self.low_dim_inputs_max.to(x.device)      # [k]

                    delta_A_pos = torch.relu(delta_A)
                    delta_A_neg = torch.relu(-delta_A)

                    lower = (
                        delta_A_pos @ z_lb
                        - delta_A_neg @ z_ub
                        + delta_b
                        + delta_mean
                        - res_drift_radius
                    )

                    upper = (
                        delta_A_pos @ z_ub
                        - delta_A_neg @ z_lb
                        + delta_b
                        + delta_mean
                        + res_drift_radius
                    )

                    int_drift_loss += self.lambda_int_input_drift * (lower.pow(2).mean() + upper.pow(2).mean())

                # Regularize all layers above
                next_layer = layers[2*idx+2]

                if (self.regularize_classifier or self.dil_mode) and hasattr(next_layer, "classifier"):
                    target_module = next_layer.classifier
                else:
                    target_module = next_layer

                if target_module is not None:
                    out_dim = next(target_module.parameters()).shape[0]
                    total_lower = torch.zeros(out_dim, device=x.device).unsqueeze(0)
                    total_upper = torch.zeros(out_dim, device=x.device).unsqueeze(0)
                  
                    for name, p in target_module.named_parameters():
                        for mod_name, mod_param in self.module.named_parameters():
                            if mod_param is p and mod_name in self.params_buffer:
                                prev_param = self.params_buffer[mod_name]
                                if "weight" in name:
                                    wd_pos = torch.relu(p - prev_param)
                                    wd_neg = torch.relu(-(p - prev_param))
                                    total_lower += (wd_pos @ lb - wd_neg @ ub)
                                    total_upper += (wd_pos @ ub - wd_neg @ lb)
                                elif "bias" in name:
                                    total_lower += (p - prev_param)
                                    total_upper += (p - prev_param)

                    int_drift_loss += self.lambda_int_hidden_drift * (total_lower.pow(2).mean() + total_upper.pow(2).mean())

        loss = (
            loss
            + self.lambda_var * var_loss
            + self.lambda_slope_reg * slope_loss
            + int_drift_loss
        )
        return loss