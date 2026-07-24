# It is a modification of https://hypnettorch.readthedocs.io/en/latest/_modules/hypnettorch/hnets/mlp_hnet.html#HMLP,
# licensed under the Apache License, Version 2.0, to enable interval bound propagation mechanism in an MLP-based hypernetwork
# when the intersections are forced.

from hypnettorch.hnets import HMLP
from hypnettorch.hnets.hnet_interface import HyperNetInterface

import torch
import torch.nn.functional as F

class HMLP_IBP(HMLP, HyperNetInterface):

    """
    Implementation of a `full hypernet` with interval bound propagation mechanism around tasks' embeddings.

    The network will consist of several hidden layers and a final linear output
    layer that produces all weight matrices/bias-vectors the network has to
    produce.

    The network allows to maintain a set of embeddings internally that can be
    used as conditional input.

    Arguments are like in https://hypnettorch.readthedocs.io/en/latest/_modules/hypnettorch/hnets/mlp_hnet.html#HMLP
    """

    def __init__(self, target_shapes, uncond_in_size=0, cond_in_size=8,
                 layers=(100, 100), verbose=True, activation_fn=torch.nn.ReLU(),
                 use_bias=True, no_uncond_weights=False, no_cond_weights=False,
                 num_cond_embs=1, dropout_rate=-1, use_spectral_norm=False,
                 use_batch_norm=False, *args, **kwargs):

        HMLP.__init__(self, target_shapes, uncond_in_size=uncond_in_size, cond_in_size=cond_in_size,
                 layers=layers, verbose=verbose, activation_fn=activation_fn,
                 use_bias=use_bias, no_uncond_weights=no_uncond_weights, no_cond_weights=no_cond_weights,
                 num_cond_embs=num_cond_embs, dropout_rate=dropout_rate, use_spectral_norm=use_spectral_norm,
                 use_batch_norm=use_batch_norm)

        
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._prev_hnet_weights = None
        
        ### Create fixed perturbation vectors
        self._perturbated_eps_T = []

        for _ in range(num_cond_embs):
            self._perturbated_eps_T.append(
                F.softmax(torch.ones(self._cond_in_size), dim=-1).to(self._device)
                )
            
        self._is_properly_setup()
    
    @property
    def perturbated_eps_T(self):
        """
        Getter method for perturbation vectors.
        """
        return self._perturbated_eps_T
    
    def detach_tensor(self, idx):
        """
        This method detaches an embedding from the computation graph.
        """
        self.conditional_params[idx].requires_grad_(False)

    def get_task_bounds(self, idx, perturbated_eps):
        """
        Helper method to compute the lower and upper bounds for a specific task embedding,
        enforcing the cosine transformation to guarantee overlapping support.
        """
        # Fetch raw embedding
        h_raw = self.conditional_params[idx]
        
        # Apply cosine transformation to bound the center
        sigma = 0.5 * perturbated_eps / self._cond_in_size
        h_cos = sigma * torch.cos(h_raw)
        
        # Calculate radius via softmax
        eps = perturbated_eps * F.softmax(self._perturbated_eps_T[idx], dim=-1)
        
        return h_cos - eps, h_cos + eps

    def get_universal_intersection(self, cond_id, perturbated_eps):
        """
        Computes the geometric intersection between the cumulative hypercube of all prior 
        tasks (0 to cond_id - 1) and the current task embedding (cond_id).
        """
        # If no prior tasks exist, just return the current task's transformed bounds
        if cond_id == 0:
            lower, upper = self.get_task_bounds(0, perturbated_eps)
            h = (upper + lower) / 2.0
            eps = (upper - lower) / 2.0
            return h.to(self._device), eps.to(self._device)

        prior_lowers = []
        prior_uppers = []
        
        # 1. Compute bounding boxes for all prior tasks (with cosine transformation applied)
        for i in range(cond_id):
            l, u = self.get_task_bounds(i, perturbated_eps)
            prior_lowers.append(l)
            prior_uppers.append(u)
            
        prior_lower_bound = torch.stack(prior_lowers).min(dim=0)[0]
        prior_upper_bound = torch.stack(prior_uppers).max(dim=0)[0]
        
        # 2. Compute bounds for the current task
        curr_lower_bound, curr_upper_bound = self.get_task_bounds(cond_id, perturbated_eps)
        
        # 3. Compute the intersection (max of lowers, min of uppers)
        intersect_lower = torch.max(prior_lower_bound, curr_lower_bound)
        intersect_upper = torch.min(prior_upper_bound, curr_upper_bound)
        
        # 4. Convert intersection bounds back to center (h) and radius (eps)
        universal_h = (intersect_upper + intersect_lower) / 2.0
        universal_eps = (intersect_upper - intersect_lower) / 2.0
        
        # Safety ReLU in case the intersection is somehow perfectly empty (negative radius)
        universal_eps = F.relu(universal_eps) 
        
        return universal_h.to(self._device), universal_eps.to(self._device)

    def forward(self, uncond_input=None, cond_input=None, cond_id=None,
                weights=None, distilled_params=None, condition=None,
                ret_format='squeezed', return_extended_output=False,
                perturbated_eps=None):
        """Compute the weights of a target network when we apply nesting."""

        uncond_input, cond_input, uncond_weights, _ = \
            self._preprocess_forward_args(uncond_input=uncond_input,
                cond_input=cond_input, cond_id=cond_id, weights=weights,
                distilled_params=distilled_params, condition=condition,
                ret_format=ret_format)

        # ---------------------------------------------------------------------
        # UNIVERSAL EMBEDDING INTERSECTION 
        # (Cosine transform is handled inside get_universal_intersection)
        # ---------------------------------------------------------------------
        if cond_id is not None:
            if not isinstance(cond_id, list):
                # Single cond_id
                h, eps = self.get_universal_intersection(cond_id, perturbated_eps)
                
                # Expand to batch size if necessary
                batch_size = 1
                if cond_input is not None:
                    batch_size = cond_input.shape[0]
                elif uncond_input is not None:
                    batch_size = uncond_input.shape[0]
                    
                if len(h.shape) == 1 or h.shape[0] != batch_size:
                    h = h.expand(batch_size, -1)
                    eps = eps.expand(batch_size, -1)
            else:
                # List of cond_ids
                h_list, eps_list = [], []
                for cid in cond_id:
                    h_c, eps_c = self.get_universal_intersection(cid, perturbated_eps)
                    h_list.append(h_c)
                    eps_list.append(eps_c)
                
                h = torch.stack(h_list, dim=0)
                eps = torch.stack(eps_list, dim=0)
        else:
            ### Fallback if cond_id is missing ###
            assert self._uncond_in_size == 0 or uncond_input is not None
            assert self._cond_in_size == 0 or cond_input is not None
            
            if uncond_input is not None:
                assert len(uncond_input.shape) == 2 and uncond_input.shape[1] == self._uncond_in_size
                h = uncond_input
            if cond_input is not None:
                assert len(cond_input.shape) == 2 and cond_input.shape[1] == self._cond_in_size
                h = cond_input
            if uncond_input is not None and cond_input is not None:
                h = torch.cat([uncond_input, cond_input], dim=1)
            
            # Apply cosine transform manually for fallback cases
            sigma = 0.5 * perturbated_eps / self._cond_in_size
            h = sigma * torch.cos(h)
                
            eps = perturbated_eps * F.softmax(torch.ones_like(h), dim=-1)
            eps = eps.to(self._device)

        ### Extract layer weights ###
        bn_scales  = []
        bn_shifts  = []
        fc_weights = []
        fc_biases  = []

        assert len(uncond_weights) == len(self.unconditional_param_shapes_ref)
        for i, idx in enumerate(self.unconditional_param_shapes_ref):
            meta = self.param_shapes_meta[idx]

            if meta['name'] == 'bn_scale':
                bn_scales.append(uncond_weights[i])
            elif meta['name'] == 'bn_shift':
                bn_shifts.append(uncond_weights[i])
            elif meta['name'] == 'weight':
                fc_weights.append(uncond_weights[i])
            else:
                assert meta['name'] == 'bias'
                fc_biases.append(uncond_weights[i])

        if not self.has_bias:
            assert len(fc_biases) == 0
            fc_biases = [None] * len(fc_weights)

        if self._use_batch_norm:
            assert len(bn_scales) == len(fc_weights) - 1

        # We proceed directly to IBP propagation. h and eps are already set up correctly.

        for i in range(len(fc_weights)):
            last_layer = i == (len(fc_weights) - 1)
            
            h = F.linear(h, fc_weights[i], bias=fc_biases[i])

            W = torch.abs(fc_weights[i])
            eps = F.linear(eps, W, bias=torch.zeros_like(fc_biases[i]))

            if not last_layer:

                if self._use_batch_norm:
                   raise Exception("BatchNorm not implemented for hypernets!")

                if self._act_fn is not None:
                    z_l, z_u = h - eps, h + eps
                    z_l, z_u = self._act_fn(z_l), self._act_fn(z_u)
                    h, eps   = (z_u + z_l) / 2, (z_u - z_l) / 2

        z_l, z_u = h-eps, h+eps

        ### Split output into target shapes ###
        ret = self._flat_to_ret_format(h, ret_format)

        if return_extended_output:
            ret_zl = self._flat_to_ret_format(z_l, ret_format)
            ret_zu = self._flat_to_ret_format(z_u, ret_format)
            radii = eps

            return ret_zl, ret, ret_zu, radii
        else:
            return ret