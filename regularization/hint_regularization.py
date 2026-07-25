# Modification of https://hypnettorch.readthedocs.io/en/latest/_modules/hypnettorch/utils/hnet_regularizer.html#calc_fix_target_reg file
# licensed under the Apache License, Version 2.0 to enable regularization of the interval hypernetwork's output in the middle and in the
# ends of interval

import torch
from hypnettorch.hnets import HyperNetInterface
from typing import List, Tuple

class IntervalHypernetRegularizer:
    """
    Regularizer for interval bound propagation in a hypernetwork.
    Maintains references to the hypernetwork and main network to compute targets
    and regularization loss across the lower, middle, and upper interval bounds.
    """

    def __init__(self, hnet: HyperNetInterface, mnet=None):
        assert isinstance(hnet, HyperNetInterface)
        self.hnet = hnet
        self.mnet = mnet

    def get_current_targets(self, task_id: int) -> Tuple[List[List[torch.Tensor]], List[List[torch.Tensor]], List[List[torch.Tensor]]]:
        """
        For all j < task_id, compute the output of the hypernetwork. This output
        will be detached from the graph before being added to the return list of
        this function.

        Note, this function sets the hypernet temporarily in eval mode. No gradients
        are computed.

        Parameters:
        -----------
            task_id: int
                The ID of the current task.

        Returns:
        --------
            An empty list if task_id is 0. Otherwise, a tuple containing lists of 
            task_id-1 lower, middle, and upper targets.
        """
        if task_id == 0:
            return [], [], []

        # We temporarily switch to eval mode for target computation (e.g., to get
        # rid of training stochasticities such as dropout).
        hnet_mode = self.hnet.training
        self.hnet.eval()

        upper_ret  = []
        middle_ret = []
        lower_ret  = []

        with torch.no_grad():
            # Get weights from previous task
            prev_weights = dict()
            uncond_params = self.hnet._prev_hnet_weights
            prev_weights['uncond_weights'] = uncond_params

            # eps is handled internally by the updated HMLP_IBP forward pass
            W_lower, W_middle, W_upper, _ = self.hnet.forward(
                cond_id=list(range(task_id)),
                ret_format='sequential',
                weights=prev_weights,
                return_extended_output=True
            )
            
            upper_ret  = [[p.detach() for p in W_tid] for W_tid in W_upper]
            middle_ret = [[p.detach() for p in W_tid] for W_tid in W_middle]
            lower_ret  = [[p.detach() for p in W_tid] for W_tid in W_lower]

        self.hnet.train(mode=hnet_mode)

        return lower_ret, middle_ret, upper_ret

    def calc_fix_target_reg(self, task_id: int, lower_targets=None, middle_targets=None, 
                            upper_targets=None, prev_theta=None, prev_task_embs=None) -> torch.Tensor:
        """
        This regularizer restricts the output-mapping for previous task embeddings.
        For all tasks j < task_id.

        Parameters:
        ------------
            task_id: int
                The ID of the current task.
            lower_targets: list, optional
                A list of outputs of the hypernetwork for the lower targets. 
            middle_targets: list, optional
                A list of outputs of the hypernetwork for the middle targets. 
            upper_targets: list, optional
                A list of outputs of the hypernetwork for the upper targets. 
            prev_theta: list, optional
            prev_task_embs: list, optional

        Returns:
        --------
            The value of the regularizer.
        """
        assert task_id > 0
        assert self.hnet.unconditional_params is not None and len(self.hnet.unconditional_params) > 0
        assert middle_targets is None or len(middle_targets) == task_id
        assert self.mnet is not None
        assert middle_targets is None or (prev_theta is None and prev_task_embs is None)
        assert prev_theta is None or prev_task_embs is not None

        # Number of tasks to be regularized.
        num_regs = task_id
        ids_to_reg = list(range(num_regs))

        assert len(self.hnet.unconditional_params) == len(self.hnet.unconditional_param_shapes)

        weights = dict()
        uncond_params = self.hnet.unconditional_params
        weights['uncond_weights'] = uncond_params

        upper_reg  = 0
        middle_reg = 0
        lower_reg  = 0

        for i in ids_to_reg:
            # eps is handled internally by the updated HMLP_IBP forward pass
            lower_weights_predicted, middle_weights_predicted, upper_weights_predicted, _ = self.hnet.forward(
                cond_id=i,
                weights=weights,
                return_extended_output=True
            )

            lower_target  = lower_targets[i]
            middle_target = middle_targets[i]
            upper_target  = upper_targets[i]
        
            # Regularize all weights of the main network.
            lower_W_target = torch.cat([w.view(-1) for w in lower_target])
            lower_W_predicted = torch.cat([w.view(-1) for w in lower_weights_predicted])

            middle_W_target = torch.cat([w.view(-1) for w in middle_target])
            middle_W_predicted = torch.cat([w.view(-1) for w in middle_weights_predicted])

            upper_W_target = torch.cat([w.view(-1) for w in upper_target])
            upper_W_predicted = torch.cat([w.view(-1) for w in upper_weights_predicted])
            
            # Loss formulation: Compute sum across dimensions, then square the result
            upper_reg_i = (upper_W_target - upper_W_predicted).sum().pow(2)
            middle_reg_i = (middle_W_target - middle_W_predicted).sum().pow(2)
            lower_reg_i = (lower_W_target - lower_W_predicted).sum().pow(2)

            upper_reg  += upper_reg_i
            middle_reg += middle_reg_i
            lower_reg  += lower_reg_i

        return (upper_reg + middle_reg + lower_reg) / (3 * num_regs)