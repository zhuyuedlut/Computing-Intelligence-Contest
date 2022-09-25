import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModel


class Bert(nn.Module):
    def __init__(self, model_dir: str):
        super(Bert, self).__init__()

        self.config = AutoConfig.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        output = self.model(input_ids=input_ids, attention_mask=attention_mask)

        return output
