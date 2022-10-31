from typing import Type, Optional, List

import os

import torch
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import AutoTokenizer, PreTrainedTokenizer

from src.datamodules.components.BaseKFoldDataModule import BaseKFoldDataModule
from src.utils.utils import read_json


class CCFDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer: PreTrainedTokenizer, is_train: bool = True):
        self.df = df
        self.title = df['title'].values
        self.assignee = df['assignee'].values
        self.abstract = df['abstract'].values
        self.tokenizer = tokenizer
        self.sep_token = tokenizer.sep_token
        self.is_train = is_train

        if self.is_train:
            self.label = df['label_id'].values

    def __len__(self):
        return len(self.title)

    def __getitem__(self, item: int):
        title = self.title[item]
        assignee = self.assignee[item]
        abstract = self.abstract[item]

        input_text = title + self.sep_token + assignee + self.sep_token + abstract
        inputs = self.tokenizer(input_text, truncation=True, max_length=400, padding='max_length')

        if self.is_train:
            label = self.label[item]

            return torch.Tensor(inputs['input_ids'], dtype=torch.long), \
                   torch.Tensor(inputs['attention_mask'], dtype=torch.long), \
                   torch.Tensor(label, dtype=torch.long)
        else:
            return torch.Tensor(inputs['input_ids'], dtype=torch.long), \
               torch.Tensor(inputs['attention_mask'], dtype=torch.long)



class CCFDataModule(BaseKFoldDataModule):
    def __init__(
        self,
        data_dir: str,
        model_dir: str,
        fold: int,
        num_class: int,
        train_batch_size: int,
        valid_batch_size: int,
        test_batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False
    ):
        super(CCFDataModule, self).__init__()

        self.save_hyperparameters()

        self.train_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.train_fold: Optional[Dataset] = None
        self.val_fold: Optional[Dataset] = None

        self.num_folds: Optional[int] = None
        self.splits: Optional[List] = None

        self.tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(self.hparams.model_dir)

    @property
    def num_class(self):
        return self.num_class

    def setup(self, stage: Optional[str] = None) -> None:
        train = pd.DataFrame(read_json(os.path.join(self.hparams.data_dir, 'train.json')))
        predict = pd.DataFrame(read_json(os.path.join(self.hparams.data_dir, 'testA.json')))

        self.train_dataset = CCFDataset(train, self.tokenizer)
        self.predict_dataset = CCFDataset(predict, self.tokenizer)

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

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(self.predict_dataset)



if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "ccf.yaml")
    cfg.data_dir = os.getenv("DATASET_PATH")
    cfg.model_dir = os.getenv("PRETRAINED_MODEL_PATH")
    _ = hydra.utils.instantiate(cfg)