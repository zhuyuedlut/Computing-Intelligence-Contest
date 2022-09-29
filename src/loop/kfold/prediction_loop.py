from typing import Any, Dict, Optional

from pytorch_lightning.loops.dataloader.prediction_loop import PredictionLoop
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.trainer.states import TrainerFn

from src.datamodules.components.BaseKFoldDataModule import BaseKFoldDataModule
from src.models.components.ensemble_voting_model import EnsembleVotingModel

class KFoldPredictionLoop(Loop):
    def __init__(self):
        super(KFoldPredictionLoop, self).__init__()
        self.prediction_loop: Optional[PredictionLoop] = None

    @property
    def done(self):
        return self.trainer.global_step >= self.trainer.max_steps

    def connect(self, prediction_loop: PredictionLoop) -> None:
        self.prediction_loop = prediction_loop

    def reset(self) -> None:
        pass

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        assert isinstance(self.trainer.datamodule, BaseKFoldDataModule)

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def advance(self, *args: Any, **kwargs: Any) -> None:
        self._reset_predicting()
        self.prediction_loop.run()

    def on_advance_end(self) -> None:
        self.replace(prediction_loop=PredictionLoop)

    def on_run_end(self) -> None:
        voting_model = EnsembleVotingModel(type(self.trainer.lightning_module), checkpoint_paths)
        voting_model.trainer = self.trainer

    def _reset_predicting(self) -> None:
        self.trainer.reset_predict_dataloader()
        self.trainer.state.fn = TrainerFn.PREDICTING
        self.trainer.predicting = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed.
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)