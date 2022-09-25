import torch
import torch.nn as nn

from transformers import AutoConfig, AutoTokenizer, AutoModel


class ClassifierNet(nn.Module):
    def __init__(
        self,
        model_dir: str,
        num_class: int,
        classifier: nn.Module,
    ):
        super(ClassifierNet, self).__init__()

        self.num_class = num_class

        self.config = AutoConfig.from_pretrained(model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir)

        self.fc = classifier(
            input_dim=self.config.hidden_size,
            num_labels=self.num_class,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)

        all_hidden_states = torch.stack(out)

