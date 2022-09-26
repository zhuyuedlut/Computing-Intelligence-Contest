import torch
import torch.nn as nn

from src.utils.model import masked_softmax, weighted_sum


class SelfAttnAggregator(nn.Module):
    def __init__(self, output_dim, attn_vector=None):
        super(SelfAttnAggregator, self).__init__()

        self.output_dim = output_dim
        if attn_vector:
            self.attn_vector = attn_vector
        else:
            self.attn_vector = nn.Linear(self.output_dim, 1)

    def forward(self, input_tensors: torch.Tensor, mask: torch.Tensor):
        self_attention_logits = self.attn_vector(input_tensors).squeeze(2)
        self_weights = masked_softmax(self_attention_logits, mask)
        input_self_attn_pooled = weighted_sum(input_tensors, self_weights)

        return input_self_attn_pooled
