from typing import Optional

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, input_dim: int, num_class: int, dropout_rate: float):
        super(Classifier, self).__init__()
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, num_class)

    def forward(self, x: torch.Tensor):
        x = self.dropout(x)

        return self.linear(x)


class MultSampleClassifier(nn.Module):
    def __init__(self, input_dim: int, num_class: int, dropout_num: int, dropout_rate: float):
        super(MultSampleClassifier, self).__init__()

        self.linear = nn.Linear(input_dim, num_class)
        self.dropout_ops = nn.ModuleList(
            [nn.Dropout(dropout_rate) for _ in range(dropout_num)]
        )

    def forward(self, x: torch.Tensor):
        logits: Optional[torch.Tensor] = None
        for i, dropout_op in enumerate(self.dropout_ops):
            if i == 0:
                out = dropout_op(x)
                logits = self.linear(out)
            else:
                temp_out = dropout_op(x)
                temp_logits = self.linear(temp_out)
                logits += temp_logits
        return logits
