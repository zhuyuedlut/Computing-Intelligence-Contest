from typing import Any, List

import torch

from torch.nn import BCEWithLogitsLoss
from torchmetrics import MeanMetric, MaxMetric
from torchmetrics.classification.accuracy import Accuracy

from pytorch_lightning import LightningModule


class CCFModule(LightningModule):
    def __init__(
            self,
            net: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
    ):
        super(CCFModule, self).__init__()

        self.save_hyperparameters(logger=False, ingore=["net"])

        self.net = net

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor) -> Any:
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        output = self.forward(x)
        logits = output.logits
        preds = torch.argmax(logits, dim=1)

        return logits, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        logits, preds, targets = self.step(batch)
        loss = BCEWithLogitsLoss(logits, targets)

        self.train_loss(loss)
        self.train_acc(preds, targets)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]) -> None:
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        logits, preds, targets = self.step(batch)
        loss = BCEWithLogitsLoss(logits, targets)

        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), prog_bar=True)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(
            optimizer=optimizer,
            num_training_steps=self.trainer.estimated_stepping_batches
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "ccf.yaml")
    _ = hydra.utils.instantiate(cfg)
