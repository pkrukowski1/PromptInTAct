import logging
from copy import deepcopy

import torch
import torch.nn as nn

from models.layers.learnable_relu import LearnableReLU
from models.layers.interval_activation import IntervalActivation

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class InTActPlusPlusLinearRegularization(nn.Module):

    def __init__(self,
            lambda_var: float = 0.01,
            lambda_slope_reg: float = 0.01,
            lambda_drift: float = 1.0,
            reduced_dim: int = 50,
        ) -> None:
        """
        Initialize the interval penalization plugin for continual learning.

        Args:
            lambda_var (float, optional): Weight of variance regularization. Default: 0.01.
            lambda_drift (float, optional): Weight of interval drift preservation for hidden layers. Default: 1.0.
            reduced_dim (int, optional): Dimension of the random projection space for input hypercubes. Default: 50.
        """
        
        super().__init__()
        self.task_id = None
        log.info(f"IntervalPenalization initialized with lambda_var={lambda_var}, "
                 f"lambda_drift={lambda_drift}, "
                 f"lambda_slope_reg={lambda_slope_reg}")

        self.lambda_var = lambda_var
        self.lambda_slope_reg = lambda_slope_reg
        self.lambda_drift = lambda_drift
        self.reduced_dim = reduced_dim

        self.projection_matrix = None
        self.low_dim_inputs_min = None
        self.low_dim_inputs_max = None

        self.input_mean = None  # Global mean (D,)
        self.num_samples = 0
        self.residual_max = None

        self.interval_act_layer: IntervalActivation = None
        self.prev_linear_layer: nn.Linear = None
        self.curr_linear_layer: nn.Linear = None
        self.learnable_relu: LearnableReLU = None

        self.use_svd_projection = None


    def setup_task(
        self,
        task_id: int,
        interval_block: nn.Sequential,
        use_svd_projection: bool = True
    ) -> None:

        self.task_id = task_id

        if task_id == 0:
            return
        
        if len(interval_block) != 3:
            raise ValueError("Interval block should consists of 3 layers: IntervalActivation, affine layer, and LearnableReLU.")
        
        self.interval_act_layer = interval_block[0]             # There is no learnable params, so we may keep the original reference
        self.curr_linear_layer = interval_block[1]              # Here we keep a reference to the original object stored in the memory
        self.prev_linear_layer = deepcopy(interval_block[1])    # Here we need a reference to the object before learning the next task
        self.learnable_relu = interval_block[2]                 # We need a reference to the original layer

        for p in self.prev_linear_layer.parameters():
            if p.requires_grad:
                p.requires_grad = False

        device = next(interval_block[1].parameters()).device
        self.use_svd_projection = use_svd_projection
        
        if self.use_svd_projection:
            # ------------------------------------------------------------
            # Phase 0: Calculate projection matrix to lower-dimensional space
            # to get hypercubes around inputs to the first layer.
            # ------------------------------------------------------------
            cls_token_repr = torch.cat([x[:, 0, :] for x in self.interval_act_layer.test_act_buffer], dim=0).to(device)

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
        # Phase 1: Forward pass over stored data
        # ------------------------------------------------------------
        learnable_relu_preacts = []
        with torch.no_grad():
            for x in cls_token_repr:
                x = x.to(device)
                x = self.prev_linear_layer(x)
                learnable_relu_preacts.append(x.detach())

        # ------------------------------------------------------------
        # Phase 2: Update activation hypercubes
        # ------------------------------------------------------------
        self.interval_act_layer.reset_range()

        # ------------------------------------------------------------
        # Phase 3: Anchor LearnableReLU hinges & activate new basis
        # ------------------------------------------------------------
        learnable_relu_preacts_all = torch.cat(learnable_relu_preacts, dim=0)
        self.learnable_relu.anchor_next_shift(
            z=learnable_relu_preacts_all,
            task_id=task_id,
            percentile=0.95,
        )
        self.learnable_relu.set_no_used_basis_functions(task_id + 1)


                    
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

        drift_loss = torch.tensor(0.0, device=x.device)
            
        # Variance regularization
        acts = self.interval_act_layer.curr_task_last_batch
        acts_flat = acts.view(acts.size(0), -1)
        var_loss = acts_flat.var(dim=0, unbiased=False).mean()

        # Slope regularization
        slope = self.learnable_relu.raw_scales[self.task_id]
        slope_loss = slope.pow(2).mean()

        if self.task_id > 0:
            
            lb = self.interval_act_layer.min.to(x.device)
            ub = self.interval_act_layer.max.to(x.device)
            
            curr_W = self.curr_linear_layer.weight        # [out, D]
            curr_b = self.curr_linear_layer.bias          # [out]

            prev_W = self.prev_linear_layer.weight
            prev_b = self.prev_linear_layer.bias

            if self.use_svd_projection:
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

                drift_loss = lower.pow(2).mean() + upper.pow(2).mean()

            else:
                out_dim = self.prev_linear_layer.weight.shape[0]
                total_lower = torch.zeros(out_dim, device=x.device).unsqueeze(0)
                total_upper = torch.zeros(out_dim, device=x.device).unsqueeze(0)

                wd_pos = torch.relu(curr_W - prev_W)
                wd_neg = torch.relu(-(curr_W - prev_W))
                total_lower += (wd_pos @ lb - wd_neg @ ub)
                total_upper += (wd_pos @ ub - wd_neg @ lb)

                total_lower += (curr_b - prev_b)
                total_upper += (curr_b - prev_b)

                drift_loss =  total_lower.pow(2).mean() + total_upper.pow(2).mean()

        loss = (
            loss
            + self.lambda_var * var_loss
            + self.lambda_slope_reg * slope_loss
            + self.lambda_drift * drift_loss
        )
        return loss