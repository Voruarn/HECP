from typing import Optional

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from catalyst.data.sampler import DistributedSamplerWrapper
from pytorch_lightning import LightningDataModule

from src.datamodule.dataset import get_dataset
from src.datamodule.dataset.base import BaseDataset
from src.message import BaseDataModuleMessage
from src.message import BaseTrainerMessage


class BaseDataModule(LightningDataModule):
    """Base class of DataModule."""

    train_dataset: BaseDataset
    val_dataset: BaseDataset
    test_dataset: BaseDataset

    def __init__(
        self,
        batch_size_per_gpu: int,
        num_workers: int,
        pin_memory: bool,
        trainer_msg: BaseTrainerMessage,
        train_dataset_cfg: Optional[DictConfig] = None,
        valid_dataset_cfg: Optional[DictConfig] = None,
        test_dataset_cfg: Optional[DictConfig] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None) -> None:
        assert stage in [None, "fit", "test"]

        self.train_dataset = \
            get_dataset(self.hparams.train_dataset_cfg.name).from_config(self.hparams.train_dataset_cfg.params)
        self.valid_dataset = \
            get_dataset(self.hparams.valid_dataset_cfg.name).from_config(self.hparams.valid_dataset_cfg.params)
        if stage == "test":
            self.test_dataset = \
                get_dataset(self.hparams.test_dataset_cfg.name).from_config(self.hparams.test_dataset_cfg.params)
    
    def get_datamodule_msg(self) -> BaseDataModuleMessage:
        num_total_devices = self.hparams.trainer_msg.num_nodes * self.hparams.trainer_msg.num_devices
        max_steps = len(self.train_dataset) // (self.hparams.batch_size_per_gpu * num_total_devices) \
            * self.hparams.trainer_msg.max_epochs
        return BaseDataModuleMessage(max_steps=max_steps)
    
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(self.train_dataset,
                            batch_size=self.hparams.batch_size_per_gpu,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            sampler=sampler)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        sampler = DistributedSampler(self.valid_dataset, shuffle=False)
        return DataLoader(self.valid_dataset,
                          batch_size=self.hparams.batch_size_per_gpu,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          sampler=sampler)
    
    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size_per_gpu,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          sampler=sampler)
