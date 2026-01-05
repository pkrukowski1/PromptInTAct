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
        self.prev_linear_layer1: nn.Linear = None
        self.curr_linear_layer1: nn.Linear = None
        self.prev_linear_layer2: nn.Linear = None
        self.curr_linear_layer2: nn.Linear = None
        self.learnable_relu: LearnableReLU = None


    def setup_task(
        self,
        task_id: int,
        mlp_block: nn.Sequential,
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
            mlp_block (nn.Sequential):
                [Linear, LearnableReLU, IntervalActivation, Linear]
        """

        self.task_id = task_id

        if task_id == 0:
            return
        
        if len(mlp_block) != 4:
            raise ValueError("Interval block should consists of 3 layers: IntervalActivation, affine layer, and LearnableReLU.")
        
        self.interval_act_layer = mlp_block[3]             # There is no learnable params, so we may keep the original reference
        
        self.curr_linear_layer1 = mlp_block[0]              # Here we keep a reference to the original object stored in the memory
        self.prev_linear_layer1 = deepcopy(mlp_block[0])    # Here we need a reference to the object before learning the next task
        
        self.curr_linear_layer2 = mlp_block[2]              # Here we keep a reference to the original object stored in the memory
        self.prev_linear_layer2 = deepcopy(mlp_block[2])    # Here we need a reference to the object before learning the next task
        
        self.learnable_relu = mlp_block[1]                 # We need a reference to the original layer

        for p in self.prev_linear_layer1.parameters():
            if p.requires_grad:
                p.requires_grad = False
        
        for p in self.prev_linear_layer2.parameters():
            if p.requires_grad:
                p.requires_grad = False

        device = next(mlp_block[0].parameters()).device
        
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
                x = self.prev_linear_layer1(x)
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

        # ============================================================
        # 1. Variance regularization (representation compactness)
        # ============================================================
        acts = self.interval_act_layer.curr_task_last_batch
        # Treat all tokens in the batch as samples for the same neurons
        acts_flat = acts.view(-1, acts.size(-1)) 
        var_loss = acts_flat.var(dim=0, unbiased=False).mean()

        # ============================================================
        # 2. Slope regularization (LearnableReLU stability)
        # ============================================================
        # Protects the basis functions of the current task from exploding
        slope = self.learnable_relu.raw_scales[self.task_id]
        slope_loss = slope.pow(2).mean()

        # ============================================================
        # 3. Drift regularization (Tasks > 0)
        # ============================================================
        if self.task_id > 0:
            # --- Layer 1: SVD Subspace Guard ---
            delta_W1 = self.curr_linear_layer1.weight - self.prev_linear_layer1.weight
            delta_b1 = self.curr_linear_layer1.bias - self.prev_linear_layer1.bias

            # Project drift into the significant subspace
            delta_A = delta_W1 @ self.projection_matrix.t()
            delta_W_res = delta_W1 - (delta_A @ self.projection_matrix)

            # Residual drift bounding
            r_max = self.residual_max.to(x.device)
            res_drift_radius = delta_W_res.abs() @ r_max 
            delta_mean = delta_W1 @ self.input_mean.to(x.device)

            z_lb, z_ub = self.low_dim_inputs_min.to(x.device), self.low_dim_inputs_max.to(x.device)
            dA_pos, dA_neg = torch.relu(delta_A), torch.relu(-delta_A)

            l1_lower = (dA_pos @ z_lb - dA_neg @ z_ub + delta_b1 + delta_mean - res_drift_radius)
            l1_upper = (dA_pos @ z_ub - dA_neg @ z_lb + delta_b1 + delta_mean + res_drift_radius)
            drift_loss += (l1_lower.pow(2).mean() + l1_upper.pow(2).mean())

            # --- Layer 2: Standard Interval Propagation ---
            # Using intervals from IntervalActivation (post-ReLU)
            delta_W2 = self.curr_linear_layer2.weight - self.prev_linear_layer2.weight
            delta_b2 = self.curr_linear_layer2.bias - self.prev_linear_layer2.bias

            z2_lb = self.interval_act_layer.min.to(x.device)
            z2_ub = self.interval_act_layer.max.to(x.device)
            dW2_pos, dW2_neg = torch.relu(delta_W2), torch.relu(-delta_W2)

            l2_lower = dW2_pos @ z2_lb - dW2_neg @ z2_ub + delta_b2
            l2_upper = dW2_pos @ z2_ub - dW2_neg @ z2_lb + delta_b2
            drift_loss += (l2_lower.pow(2).mean() + l2_upper.pow(2).mean())

        # Final Weighted Loss
        return (
            loss
            + self.lambda_var * var_loss
            + self.lambda_slope_reg * slope_loss
            + self.lambda_drift * drift_loss
        )