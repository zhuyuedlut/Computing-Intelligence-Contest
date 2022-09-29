import os

from typing import List, Optional

import torch
import torch.nn as nn

from torch.nn import ModuleList

from transformers import AutoConfig, AutoModel

from src.models.components.aggregator.aggregator_layer import AggregatorLayer


class Bert(nn.Module):
    def __init__(
        self,
        model_dir: str,
        aggregator_names: List = []
    ):
        super(Bert, self).__init__()

        self.config = AutoConfig.from_pretrained(model_dir)
        self.model = AutoModel.from_pretrained(model_dir)

        self.aggregator_names = aggregator_names

        self.aggregators: ModuleList = nn.ModuleList()

        for aggregator_name in self.aggregator_names:
            if aggregator_name == 'bert_pooler':
                continue
            else:
                aggregator_op = AggregatorLayer(hidden_size=self.config.hidden_size, aggregator_name=aggregator_name)
                self.aggregators.append(aggregator_op)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output, bert_pooled_output = outputs[0], outputs[1]

        list_pooled_outputs: List[Optional[torch.Tensor]] = []

        if "bert_pooler" in self.aggregator_names:
            list_pooled_outputs.append(bert_pooled_output)

        for aggre_op in self.aggregators:
            pooled_outputs_ = aggre_op(sequence_output, mask=attention_mask)
            list_pooled_outputs.append(pooled_outputs_)

        pooled_outputs = sum(list_pooled_outputs)

        return pooled_outputs

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "ccf.yaml")
    _ = hydra.utils.instantiate(cfg)