from models.layers.interval_activation import IntervalActivation

import torch.nn as nn

def detach_interval_last_batches(curr_classifier_head: nn.Sequential) -> None:
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