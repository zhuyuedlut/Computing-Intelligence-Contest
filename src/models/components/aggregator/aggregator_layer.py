from typing import Optional

import torch
import torch.nn as nn

from src.models.components.aggregator.max_pooler import MaxPoolerAggregator
from src.models.components.aggregator.avg_pool import AvgPoolerAggregator
from src.models.components.aggregator.self_attn_pool import SelfAttnAggregator


class AggregatorLayer(nn.Module):
    def __init__(self, hidden_size: int, aggregator_name: str):
        super(AggregatorLayer, self).__init__()

        self.hidden_size = hidden_size
        self.aggregator_name = aggregator_name

        self.aggregator_op: Optional[nn.Module] = None

        if self.aggregator_name == 'slf_attn_pooler':
            attn_vector = nn.Linear(self.hidden_size, 1)
            self.aggregator_op = SelfAttnAggregator(self.hidden_size, attn_vector=attn_vector)
        elif self.aggregator_name == 'max_pooler':
            self.aggregator_op = MaxPoolerAggregator()
        else:
            self.aggregator_op = AvgPoolerAggregator()

    def forward(self, input_tensors: torch.Tensor, mask: torch.Tensor):
        output = self.aggregator_op(input_tensors, mask)

        return output