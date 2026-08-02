# train.py 第一行
import os
import sys

# 强制禁用所有多线程和多进程
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'


import hydra
from omegaconf import OmegaConf
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning import callbacks
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import TensorBoardLogger

from src.module import get_module
from src.datamodule import get_datamodule
from src.utils.utils import get_pylogger
from src.message import BaseTrainerMessage

import torch
torch.set_float32_matmul_precision('high')
torch.autograd.set_detect_anomaly(True)

def get_callbacks(callback_cfg: DictConfig) -> list:
    _callbacks = []
    for k, v in callback_cfg.items():
        # _callbacks.append(eval(f"callbacks.{k}")(**v))
        # 修改后 —— 支持 "ClassName_suffix" 命名约定
        class_name = k.split('_', 1)[0]
        _callbacks.append(eval(f"callbacks.{class_name}")(**v))
    return _callbacks

def get_trainer_msg(trainer: Trainer) -> BaseTrainerMessage:
    return BaseTrainerMessage(
        max_epochs=trainer.max_epochs,
        num_nodes=trainer.num_nodes,
        num_devices=trainer.num_devices,
    )


@hydra.main(config_path="config", config_name="train", version_base="1.2")
def train(cfg: DictConfig) -> None:
    logger = get_pylogger(__name__)
    
    logger.info(OmegaConf.to_yaml(cfg))

    # Fix seed via Pytorch Lightning.
    # Set trainer.deterministic as True for reproducibility.
    seed = cfg.experiment.get("seed")
    if seed is not None:
        seed_everything(seed, workers=True)

    logger.info("Instantiating callbacks...")
    callbacks = get_callbacks(cfg.callback)

    logger.info("Instantiating tensorboard logger...")
    tb_logger = TensorBoardLogger(
        save_dir=cfg.experiment.work_dir,
        name=cfg.experiment.name,
        sub_dir="tensorboard",
        default_hp_metric=False,
    )

    logger.info("Instantiating trainer...")

    # trainer = Trainer(
    #     logger=tb_logger,
    #     callbacks=callbacks,
    #     **cfg.trainer,
    # )
    trainer_cfg = OmegaConf.to_container(cfg.trainer, resolve=True)
    trainer_cfg.pop('hydra', None)

    # # 强制禁用所有分布式策略
    # trainer_cfg['strategy'] = 'auto'
    # trainer_cfg['devices'] = 1  # 只使用1张GPU
    # trainer_cfg['num_nodes'] = 1
    # trainer_cfg['accelerator'] = 'gpu'

    # # 禁用所有多进程相关功能
    # trainer_cfg['enable_progress_bar'] = True
    # trainer_cfg['sync_batchnorm'] = False

    # 进度条和日志优化配置
    trainer_cfg.update({
        'enable_progress_bar': True,
        'enable_model_summary': False,  # 关闭启动时冗长的模型参数表
        'log_every_n_steps': 1,         # 进度条每1步更新一次loss
        'accelerator': 'gpu',
        'devices': 1,
        'strategy': 'auto',
        'num_nodes': 1,
        'sync_batchnorm': False
    })

    trainer = Trainer(
        logger=tb_logger,
        callbacks=callbacks,
        **trainer_cfg,
    )

    trainer_msg = get_trainer_msg(trainer=trainer)

    logger.info(f"Instantiating datamodule: {cfg.datamodule.name}")
    datamodule = get_datamodule(cfg.datamodule.name)(
        **cfg.datamodule.params,
        trainer_msg=trainer_msg,
        train_dataset_cfg=cfg.train_dataset,
        valid_dataset_cfg=cfg.valid_dataset,
    )
    datamodule.setup(stage="fit")
    datamodule_msg = datamodule.get_datamodule_msg()

    logger.info(f"Instantiating module: {cfg.module.name}")
    module = get_module(cfg.module.name)(
        train_loss_cfg=cfg.train_loss,
        valid_loss_cfg=cfg.valid_loss,
        metric_cfg=cfg.metric,
        model_cfg=cfg.model,
        optimizer_cfg=cfg.optimizer,
        scheduler_cfg=cfg.scheduler,
        datamodule_msg=datamodule_msg,
    )

    trainer.fit(model=module, datamodule=datamodule)


if __name__ == "__main__":
    train()
