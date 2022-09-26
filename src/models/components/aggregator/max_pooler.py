import torch
import torch.nn as nn

from src.utils.model import replace_masked_values


class MaxPoolerAggregator(nn.Module):
    def __init__(self):
        super(MaxPoolerAggregator, self).__init__()

    def forward(self, input_tensor: torch.Tensor, mask: torch.Tensor):
        if mask is not None:
            input_tensor = replace_masked_values(input_tensor, mask.unsqueeze(2), -1e7)
        input_max_pooled = torch.max(input_tensor, 1)[0]

        return input_max_pooled
