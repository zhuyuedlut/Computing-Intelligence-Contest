import torch
import torch.nn as nn


class ClassifierNet(nn.Module):
    def __init__(
        self,
        model_dir: str,
        num_class: int,
        bert: nn.Module,
        classifier: nn.Module,
    ):
        super(ClassifierNet, self).__init__()

        self.num_class = num_class

        self.fc = classifier(
            input_dim=self.config.hidden_size,
            num_labels=self.num_class,
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        pooled_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        logits = self.fc(pooled_outputs)

        return logits

