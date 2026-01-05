import logging
from copy import deepcopy

import torch
import torch.nn as nn

from models.layers.learnable_relu import LearnableReLU
from models.layers.interval_activation import IntervalActivation

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

class InTActPlusPlusMlpBlockRegularization(nn.Module):
    """
    InTAct++ Linear Regularization Module for Continual Learning.

    This module implements a *functional drift regularizer* that constrains how
    much a linear layer's output can change across tasks. It combines:

    - Interval Arithmetic (IA) for worst-case drift bounds
    - SVD-based low-dimensional subspace projection
    - Residual drift bounding for discarded dimensions
    - Variance regularization for representation compactness
    - Slope regularization for LearnableReLU stability

    The regularizer is designed to be applied as an *augmentation to the task loss*
    during training and assumes the following architectural block:

        IntervalActivation -> Linear -> LearnableReLU
    """
    def __init__(self,
            lambda_var: float = 0.01,
            lambda_slope_reg: float = 0.01,
            lambda_drift: float = 1.0,
            reduced_dim: int = 50,
        ) -> None:
        """
        Initialize the InTAct++ regularizer.

        Args:
            lambda_var (float): Weight for activation variance regularization.
            lambda_slope_reg (float): Weight for LearnableReLU slope regularization.
            lambda_drift (float): Weight for functional drift penalty.
            reduced_dim (int): Dimensionality of the SVD projection subspace.
        """
        
        super().__init__()
        self.task_id = None
        log.info(
            f"InTAct++ initialized with "
            f"lambda_var={lambda_var}, "
            f"lambda_slope_reg={lambda_slope_reg}, "
            f"lambda_drift={lambda_drift}"
        )

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


    def setup_task(
        self,
        task_id: int,
        interval_block: nn.Sequential,
    ) -> None:
        """
        Prepare the regularizer for a new task.

        This method:
        - Freezes a copy of the previous linear layer
        - Builds or updates the SVD projection basis
        - Updates low-dimensional input bounds
        - Anchors LearnableReLU hinges
        - Resets interval activation ranges

        Args:
            task_id (int): Index of the current task.
            interval_block (nn.Sequential):
                [IntervalActivation, Linear, LearnableReLU]
        """

        self.task_id = task_id

        if task_id == 0:
            return
        
        if len(interval_block) != 3:
            raise ValueError("Interval block should consists of 3 layers: IntervalActivation, affine layer, and LearnableReLU.")
        
        self.interval_act_layer = interval_block[2]             # There is no learnable params, so we may keep the original reference
        self.curr_linear_layer = interval_block[0]              # Here we keep a reference to the original object stored in the memory
        self.prev_linear_layer = deepcopy(interval_block[0])    # Here we need a reference to the object before learning the next task
        self.learnable_relu = interval_block[1]                 # We need a reference to the original layer

        for p in self.prev_linear_layer.parameters():
            if p.requires_grad:
                p.requires_grad = False

        device = next(interval_block[1].parameters()).device
        
        # ============================================================
        # Phase 0 — Input Subspace Construction (SVD)
        # ============================================================
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

        # ============================================================
        # Phase 1 — Replay through previous linear layer
        # ============================================================
        learnable_relu_preacts = []
        with torch.no_grad():
            for x in cls_token_repr:
                x = x.to(device)
                x = self.prev_linear_layer(x)
                learnable_relu_preacts.append(x.detach())

        # ============================================================
        # Phase 2 — Reset activation intervals
        # ============================================================
        self.interval_act_layer.reset_range()

        # ============================================================
        # Phase 3 — Anchor LearnableReLU hinges
        # ============================================================
        learnable_relu_preacts_all = torch.cat(learnable_relu_preacts, dim=0)
        self.learnable_relu.anchor_next_shift(
            z=learnable_relu_preacts_all,
            task_id=task_id,
            percentile=0.95,
        )
        self.learnable_relu.set_no_used_basis_functions(task_id + 1)


                    
    def forward(self, x: torch.Tensor, loss: torch.Tensor) -> torch.Tensor:
        """
        Augment task loss with InTAct++ regularization terms.

        Regularization components:
            - Activation variance minimization
            - LearnableReLU slope regularization
            - Functional drift penalty (interval-based)

        Args:
            x (torch.Tensor): Input batch.
            loss (torch.Tensor): Task loss.

        Returns:
            torch.Tensor: Total loss.
        """

        drift_loss = torch.tensor(0.0, device=x.device)
            
        # ============================================================
        # Variance regularization (representation compactness)
        # ============================================================
        acts = self.interval_act_layer.curr_task_last_batch
        acts_flat = acts.view(acts.size(0), -1)
        var_loss = acts_flat.var(dim=0, unbiased=False).mean()

        # ============================================================
        # Slope regularization (LearnableReLU stability)
        # ============================================================
        slope = self.learnable_relu.raw_scales[self.task_id]
        slope_loss = slope.pow(2).mean()


        # ============================================================
        # Drift regularization (tasks > 0)
        # ============================================================
        if self.task_id > 0:
            
            lb = self.interval_act_layer.min.to(x.device)
            ub = self.interval_act_layer.max.to(x.device)
            
            curr_W = self.curr_linear_layer.weight        # [out, D]
            curr_b = self.curr_linear_layer.bias          # [out]

            prev_W = self.prev_linear_layer.weight
            prev_b = self.prev_linear_layer.bias

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

        loss = (
            loss
            + self.lambda_var * var_loss
            + self.lambda_slope_reg * slope_loss
            + self.lambda_drift * drift_loss
        )
        return loss