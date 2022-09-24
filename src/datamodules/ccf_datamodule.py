from typing import Type, Optional, List

import os

import torch
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, PreTrainedTokenizer

from components.BaseKFoldDataModule import BaseKFoldDataModule
from src.utils.utils import read_json


class CCFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer):
        self.df = df
        self.title = df['title'].values
        self.assignee = df['assignee'].values
        self.abstract = df['abstract'].values
        self.label = df['label_id'].values
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token

    def __len__(self):
        return len(self.title)

    def __getitem__(self, item: int):
        label = int(self.label[item])
        title = self.title[item]
        assignee = self.assignee[item]
        abstract = self.abstract[item]

        input_text = title + self.sep_token + assignee + self.sep_token + abstract

        inputs = self.tokenizer(input_text, truncation=True, max_length=400, padding='max_length')
        return torch.Tensor(inputs['input_ids'], dtype=torch.long), \
               torch.Tensor(inputs['attention_mask'], dtype=torch.long), \
               torch.Tensor(label, dtype=torch.long)


@dataclass
class CCFDataModule(BaseKFoldDataModule):
    data_dir: str
    model_dir: str
    fold: int
    train_batch_size: int
    valid_batch_size: int
    test_batch_size: int
    num_workers: int = 0
    pin_memory: bool = False

    @property
    def num_class(self):
        return 36

    def setup(self, stage: Optional[str] = None) -> None:
        train = pd.DataFrame(read_json(os.path.join(self.hparams.data_dir, 'train.json')))
        test = pd.DataFrame(read_json(os.path.join(self.hparams.data_dir, 'testA.json')))

        self.train_dataset = CCFDataset(train, self.tokenizer)
        self.test_dataset = CCFDataset(test, self.tokenizer)

    def setup_folds(self, num_folds: int) -> None:
        self.num_folds = num_folds
        self.splits = [
            split for split in
            StratifiedKFold(n_splits=self.fold).split(X=self.train_dataset.df, y=self.train_dataset.label,
                                                      groups=self.train_dataset.label)]
        print(self.splits)

    def setup_fold_index(self, fold_index: int) -> None:
        train_indices, val_indices = self.splits[fold_index]
        self.train_fold = Subset(self.train_dataset, train_indices)
        self.val_fold = Subset(self.train_dataset, val_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_fold)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset)

    def __post_init__(self):
        super(CCFDataModule, self).__init__()

        self.save_hyperparameters()

        self.train_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.train_fold: Optional[Dataset] = None
        self.val_fold: Optional[Dataset] = None

        self.num_folds: Optional[int] = None
        self.splits: Optional[List] = None

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.hparams.model_dir)


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "ccf.yaml")
    cfg.data_dir = os.getenv("DATASET_PATH")
    _ = hydra.utils.instantiate(cfg)