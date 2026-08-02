import time
from typing import Optional

import torch_optimizer as optim
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from src.message import BaseDataModuleMessage
from src.model import get_model
from src.module.optimizer import get_optimizer
from src.module.optimizer.scheduler import get_scheduler
from src.utils.utils import get_pylogger


class BaseModule(LightningModule):
    def __init__(
        self,
        model_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        datamodule_msg: Optional[BaseDataModuleMessage],
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = get_model(model_cfg.name)(
            **model_cfg.params,
        )

        self.optimizer_cfg = optimizer_cfg
        self.scheduler_cfg = scheduler_cfg

        self.custom_logger = get_pylogger(self.__class__.__name__)

    def configure_optimizers(self) -> dict:
        no_decay = ["bias", "bn"]

        grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.optimizer_cfg.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        optimizer = get_optimizer(self.hparams.optimizer_cfg.name)(
            grouped_parameters,
            **self.hparams.optimizer_cfg.params,
        )
        if self.hparams.optimizer_cfg.lookahead.active:
            optimizer = optim.Lookahead(optimizer, **self.hparams.optimizer_cfg.lookahead.params)

        scheduler = get_scheduler(
            optimizer=optimizer,
            scheduler_cfg=self.hparams.scheduler_cfg,
            max_steps=self.hparams.datamodule_msg.max_steps,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

    def on_train_epoch_start(self) -> None:
        msg = (
            f"[Epoch: {self.current_epoch + 1}/{self.trainer.max_epochs}] [Train Epoch Start] "
        )
        self.custom_logger.info(msg)
        self.train_epoch_start = time.time()

    def on_train_epoch_end(self) -> None:
        msg = (
            f"[Epoch: {self.current_epoch + 1}/{self.trainer.max_epochs}] "
            f"[Train Epoch End: {(time.time() - self.train_epoch_start):.4f} Sec/Epoch]"
        )
        self.custom_logger.info(msg)

    def on_train_batch_end(self, *args, **kwargs) -> None:
        pass
