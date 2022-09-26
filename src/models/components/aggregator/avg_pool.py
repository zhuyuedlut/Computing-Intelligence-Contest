import torch
import torch.nn as nn

from src.utils.model import replace_masked_values


class AvgPoolerAggregator(nn.Module):
    def __init__(self):
        super(AvgPoolerAggregator, self).__init__()

    def forward(self, input_tensor: torch.Tensor, mask: torch.Tensor):
        if mask is not None:
            input_tensor = replace_masked_values(input_tensor, mask.unsqueeze(2), 0)

        token_avg_pooled = torch.mean(input_tensor, 1)
        return token_avg_pooled
