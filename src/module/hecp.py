from typing import List
from typing import Optional

import torch
from omegaconf import DictConfig
from torchmetrics import MeanMetric
from PIL import Image
from loguru import logger

from src.message import BaseDataModuleMessage
from src.module import register_module
from src.module.base import BaseModule
from src.module.loss import get_loss
from src.module.metric import get_metric


@register_module(name="hecp")
class HallocModule(BaseModule):
    def __init__(
        self,
        train_loss_cfg: DictConfig,
        valid_loss_cfg: DictConfig,
        metric_cfg: DictConfig,
        model_cfg: DictConfig,
        optimizer_cfg: DictConfig,
        scheduler_cfg: DictConfig,
        datamodule_msg: Optional[BaseDataModuleMessage],
    ) -> None:
        super().__init__(
            model_cfg=model_cfg,
            optimizer_cfg=optimizer_cfg,
            scheduler_cfg=scheduler_cfg,
            datamodule_msg=datamodule_msg,
        )

        self.obj_metric = get_metric(metric_cfg.name)(classes=[0, 1], **metric_cfg.params)
        self.att_metric = get_metric(metric_cfg.name)(classes=[0, 1], **metric_cfg.params)
        self.rel_metric = get_metric(metric_cfg.name)(classes=[0, 1], **metric_cfg.params)
        self.sce_metric = get_metric(metric_cfg.name)(classes=[0, 1], **metric_cfg.params)
        self.oth_metric = get_metric(metric_cfg.name)(classes=[0, 1], **metric_cfg.params)
        self.all_metric = get_metric(metric_cfg.name)(classes=[0, 1], **metric_cfg.params)
        self.total_metric = get_metric(metric_cfg.name)(classes=[0, 1], **metric_cfg.params)

        self.train_obj_loss = self.initialize_loss(train_loss_cfg, weight=[1.0, 1.0])
        self.train_att_loss = self.initialize_loss(train_loss_cfg, weight=[1.0, 1.0])
        self.train_rel_loss = self.initialize_loss(train_loss_cfg, weight=[1.0, 1.0])
        self.train_sce_loss = self.initialize_loss(train_loss_cfg, weight=[1.0, 1.0])
        self.train_oth_loss = self.initialize_loss(train_loss_cfg, weight=[1.0, 1.0])
        self.train_all_loss = self.initialize_loss(train_loss_cfg, weight=[1.0, 1.0])

        self.train_obj_loss_metric = MeanMetric()
        self.train_att_loss_metric = MeanMetric()
        self.train_rel_loss_metric = MeanMetric()
        self.train_sce_loss_metric = MeanMetric()
        self.train_oth_loss_metric = MeanMetric()
        self.train_all_loss_metric = MeanMetric()
        self.train_total_loss_metric = MeanMetric()

        self.valid_obj_loss = self.initialize_loss(valid_loss_cfg, weight=[1.0, 1.0])
        self.valid_att_loss = self.initialize_loss(valid_loss_cfg, weight=[1.0, 1.0])
        self.valid_rel_loss = self.initialize_loss(valid_loss_cfg, weight=[1.0, 1.0])
        self.valid_sce_loss = self.initialize_loss(valid_loss_cfg, weight=[1.0, 1.0])
        self.valid_oth_loss = self.initialize_loss(valid_loss_cfg, weight=[1.0, 1.0])
        self.valid_all_loss = self.initialize_loss(valid_loss_cfg, weight=[1.0, 1.0])
        
        self.valid_obj_loss_metric = MeanMetric()
        self.valid_att_loss_metric = MeanMetric()
        self.valid_rel_loss_metric = MeanMetric()
        self.valid_sce_loss_metric = MeanMetric()
        self.valid_oth_loss_metric = MeanMetric()
        self.valid_all_loss_metric = MeanMetric()
        self.valid_total_loss_metric = MeanMetric()

        self.is_all = False

    def initialize_loss(self, loss_cfg: DictConfig, weight: List[int]):
        return get_loss(loss_cfg.name)(
            weight=torch.tensor(weight).to(self.device),
            size_average=loss_cfg.params.size_average,
            ignore_index=loss_cfg.params.ignore_index,
            reduce=loss_cfg.params.reduce,
            reduction=loss_cfg.params.reduction,
            label_smoothing=loss_cfg.params.label_smoothing,
        )

    def training_step(self, batch: dict, batch_idx: int, *args, **kwargs) -> dict:
        # print("Batch available keys:", list(batch.keys()))
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        embeddings = batch["hidden_states"]

        attention_masks = batch["attention_masks"]
        image_paths = batch["image_paths"]
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

        if "all_labels" in batch.keys():
           # logger.info("all_labels in batch.keys()")

            self.is_all = True
            all_labels = batch["all_labels"]
            
            all_logits = self.model(
                input_ids = input_ids,
                embeddings = embeddings,
                attention_masks=attention_masks,
                token_type_ids = token_type_ids,
                images=images,
                is_all=self.is_all,
            )
            
            attention_masks = attention_masks.view(-1)  # (B * L,)
            nonzero_indices = attention_masks.nonzero(as_tuple=True)

            all_logits = all_logits.view(-1, all_logits.size(-1))  # (B * L, C)
            all_labels = all_labels.view(-1)
            filtered_all_logits = all_logits[nonzero_indices]
            filtered_all_probs = torch.softmax(filtered_all_logits, dim=-1)
            filtered_all_labels = all_labels[nonzero_indices]

            loss = self.train_all_loss(filtered_all_logits, filtered_all_labels)

            self.all_metric.update(filtered_all_probs, filtered_all_labels, is_train=True)
            self.total_metric.update(filtered_all_probs, filtered_all_labels, is_train=True)

            self.train_all_loss_metric.update(loss)
            self.train_total_loss_metric.update(loss)
           
            self.log(
                "train_loss", 
                loss, 
                prog_bar=True,  # 显示在进度条中
                logger=True,    # 记录到TensorBoard
                on_step=True,   # 每step更新
                on_epoch=True,  # 自动计算epoch平均
            )

            return {
                "loss": loss,
                "all_loss": loss,
            }
        else:
            self.is_all = False

            obj_labels = batch["obj_labels"]
            att_labels = batch["att_labels"]
            rel_labels = batch["rel_labels"]
            sce_labels = batch["sce_labels"]
            oth_labels = batch["oth_labels"]

            obj_logits, att_logits, rel_logits, sce_logits, oth_logits = self.model(
                input_ids = input_ids,
                embeddings = embeddings,
                attention_masks=attention_masks,
                token_type_ids = token_type_ids,
                images=images,
                is_all=self.is_all,
            )
            
            # logger.info(f"embeddings.shape: {embeddings.shape}")
            # logger.info(f"attention_masks.shape: {attention_masks.shape}")
            # logger.info(f"obj_logits.shape: {obj_logits.shape}")
            # logger.info(f"att_logits.shape: {att_logits.shape}")
            # logger.info(f"rel_logits.shape: {rel_logits.shape}")
            # logger.info(f"sce_logits.shape: {sce_logits.shape}")
            # logger.info(f"oth_logits.shape: {oth_logits.shape}")
            # logger.info(f"obj_labels.shape: {obj_labels.shape}")
            # logger.info(f"att_labels.shape: {att_labels.shape}")
            # logger.info(f"rel_labels.shape: {rel_labels.shape}")
            # logger.info(f"sce_labels.shape: {sce_labels.shape}")
            # logger.info(f"oth_labels.shape: {oth_labels.shape}")
            # logger.info(f"obj_logits[:10]: {obj_logits[:10]}")
            # logger.info(f"att_logits[:10]: {att_logits[:10]}")
            # logger.info(f"rel_logits[:10]: {rel_logits[:10]}")
            # logger.info(f"sce_logits[:10]: {sce_logits[:10]}")
            # logger.info(f"oth_logits[:10]: {oth_logits[:10]}")
            # logger.info(f"obj_labels[:10]: {obj_labels[:10]}")
            # logger.info(f"att_labels[:10]: {att_labels[:10]}")
            # logger.info(f"rel_labels[:10]: {rel_labels[:10]}")
            # logger.info(f"sce_labels[:10]: {sce_labels[:10]}")
            # logger.info(f"oth_labels[:10]: {oth_labels[:10]}")

            attention_masks = attention_masks.view(-1)  # (B * L,)
            nonzero_indices = attention_masks.nonzero(as_tuple=True)

            obj_logits = obj_logits.view(-1, obj_logits.size(-1))  # (B * L, C)
            obj_labels = obj_labels.view(-1)  # (B * L)
            filtered_obj_logits = obj_logits[nonzero_indices]
            filtered_obj_probs = torch.softmax(filtered_obj_logits, dim=-1)
            filtered_obj_labels = obj_labels[nonzero_indices]
            obj_loss = self.train_obj_loss(filtered_obj_logits, filtered_obj_labels)

            att_logits = att_logits.view(-1, att_logits.size(-1))  # (B * L, C)
            att_labels = att_labels.view(-1)  # (B * L)
            filtered_att_logits = att_logits[nonzero_indices]
            filtered_att_probs = torch.softmax(filtered_att_logits, dim=-1)
            filtered_att_labels = att_labels[nonzero_indices]
            att_loss = self.train_att_loss(filtered_att_logits, filtered_att_labels)

            rel_logits = rel_logits.view(-1, rel_logits.size(-1))  # (B * L, C)
            rel_labels = rel_labels.view(-1)  # (B * L)
            filtered_rel_logits = rel_logits[nonzero_indices]
            filtered_rel_probs = torch.softmax(filtered_rel_logits, dim=-1)
            filtered_rel_labels = rel_labels[nonzero_indices]
            rel_loss = self.train_rel_loss(filtered_rel_logits, filtered_rel_labels)

            sce_logits = sce_logits.view(-1, sce_logits.size(-1))  # (B * L, C)
            sce_labels = sce_labels.view(-1)  # (B * L)
            filtered_sce_logits = sce_logits[nonzero_indices]
            filtered_sce_probs = torch.softmax(filtered_sce_logits, dim=-1)
            filtered_sce_labels = sce_labels[nonzero_indices]
            sce_loss = self.train_sce_loss(filtered_sce_logits, filtered_sce_labels)

            oth_logits = oth_logits.view(-1, oth_logits.size(-1))  # (B * L, C)
            oth_labels = oth_labels.view(-1)  # (B * L)
            filtered_oth_logits = oth_logits[nonzero_indices]
            filtered_oth_probs = torch.softmax(filtered_oth_logits, dim=-1)
            filtered_oth_labels = oth_labels[nonzero_indices]
            oth_loss = self.train_oth_loss(filtered_oth_logits, filtered_oth_labels)

            total_probs = torch.max(filtered_obj_probs[:, 1], filtered_att_probs[:, 1])
            total_probs = torch.max(total_probs, filtered_rel_probs[:, 1])
            total_probs = torch.max(total_probs, filtered_sce_probs[:, 1])
            total_probs = torch.max(total_probs, filtered_oth_probs[:, 1])
            total_probs = torch.stack([1 - total_probs, total_probs], dim=-1)
            total_labels = (filtered_obj_labels | filtered_att_labels | filtered_rel_labels | filtered_sce_labels | filtered_oth_labels)

            loss = obj_loss + att_loss + rel_loss + sce_loss + oth_loss

           # logger.info(f"train loss: {loss}")
            self.log(
                "train_loss", 
                loss, 
                prog_bar=True,  # 显示在进度条中
                logger=True,    # 记录到TensorBoard
                on_step=True,   # 每step更新
                on_epoch=True,  # 自动计算epoch平均
            )
            
            self.obj_metric.update(filtered_obj_probs, filtered_obj_labels, is_train=True)
            self.att_metric.update(filtered_att_probs, filtered_att_labels, is_train=True)
            self.rel_metric.update(filtered_rel_probs, filtered_rel_labels, is_train=True)
            self.sce_metric.update(filtered_sce_probs, filtered_sce_labels, is_train=True)
            self.oth_metric.update(filtered_oth_probs, filtered_oth_labels, is_train=True)
            self.total_metric.update(total_probs, total_labels, is_train=True)

            self.train_obj_loss_metric.update(obj_loss)
            self.train_att_loss_metric.update(att_loss)
            self.train_rel_loss_metric.update(rel_loss)
            self.train_sce_loss_metric.update(sce_loss)
            self.train_oth_loss_metric.update(oth_loss)
            self.train_total_loss_metric.update(loss)

            
            return {
                "loss": loss,
                "obj_loss": obj_loss,
                "att_loss": att_loss,
                "rel_loss": rel_loss,
                "sce_loss": sce_loss,
                "oth_loss": oth_loss,
            }

    def on_training_epoch_end(self) -> None:
        loss = self.train_total_loss_metric.compute()
        self.train_total_loss_metric.reset()
        
        # 获取当前epoch的平均train_loss
        avg_loss = self.trainer.callback_metrics.get('train_loss_epoch', 0.0)
        # 每个epoch结束时打印一次
        self.logger.info(f"Epoch {self.current_epoch} finished | Average train loss: {avg_loss:.6f}")
        
        metric_msg = []
        self.log("train_loss", loss, prog_bar=True, logger=False, sync_dist=True)
        self.logger.log_metrics(
            metrics={
                "train_loss": loss,
            },
            step=self.current_epoch,
        )
        metric_msg.append(f"train_loss: {loss:.4f}")

        if self.is_all:
            for k, v in self.all_metric.compute(is_train=True).items():
                k = f"all/{k}"
                self.log(k, v, logger=False, prog_bar=False, sync_dist=True)
                self.logger.log_metrics(
                    metrics={
                        k: v,
                    },
                    step=self.current_epoch,
                )
                metric_msg.append(f"{k}: {v:.4f}")
        else:
            for k, v in self.obj_metric.compute(is_train=True).items():
                k = f"obj/{k}"
                self.log(k, v, logger=False, prog_bar=False, sync_dist=True)
                self.logger.log_metrics(
                    metrics={
                        k: v,
                    },
                    step=self.current_epoch,
                )
                metric_msg.append(f"{k}: {v:.4f}")

            for k, v in self.att_metric.compute(is_train=True).items():
                k = f"att/{k}"
                self.log(k, v, logger=False, prog_bar=False, sync_dist=True)
                self.logger.log_metrics(
                    metrics={
                        k: v,
                    },
                    step=self.current_epoch,
                )
                metric_msg.append(f"{k}: {v:.4f}")

            for k, v in self.rel_metric.compute(is_train=True).items():
                k = f"rel/{k}"
                self.log(k, v, logger=False, prog_bar=False, sync_dist=True)
                self.logger.log_metrics(
                    metrics={
                        k: v,
                    },
                    step=self.current_epoch,
                )
                metric_msg.append(f"{k}: {v:.4f}")

            for k, v in self.sce_metric.compute(is_train=True).items():
                k = f"sce/{k}"
                self.log(k, v, logger=False, prog_bar=False, sync_dist=True)
                self.logger.log_metrics(
                    metrics={
                        k: v,
                    },
                    step=self.current_epoch,
                )
                metric_msg.append(f"{k}: {v:.4f}")

            for k, v in self.oth_metric.compute(is_train=True).items():
                k = f"oth/{k}"
                self.log(k, v, logger=False, prog_bar=False, sync_dist=True)
                self.logger.log_metrics(
                    metrics={
                        k: v,
                    },
                    step=self.current_epoch,
                )
                metric_msg.append(f"{k}: {v:.4f}")

            for k, v in self.total_metric.compute(is_train=True).items():
                k = f"total/{k}"
                self.log(k, v, logger=False, prog_bar=False, sync_dist=True)
                self.logger.log_metrics(
                    metrics={
                        k: v,
                    },
                    step=self.current_epoch,
                )
                metric_msg.append(f"{k}: {v:.4f}")

        metric_msg = "\n".join(metric_msg)
        msg = (
            f"[Epoch: {self.current_epoch + 1}/{self.trainer.max_epochs}] [Train Evaluation]\n"
            f"{metric_msg}"
        )
        self.custom_logger.info(msg)

    def validation_step(self, batch: dict, batch_idx: int, *args, **kwargs) -> dict:
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        embeddings = batch["hidden_states"]

        attention_masks = batch["attention_masks"]
        image_paths = batch["image_paths"]
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        
        if "all_labels" in batch.keys():
            self.is_all = True
            all_labels = batch["all_labels"]

            all_logits = self.model(
                input_ids = input_ids,
                embeddings = embeddings,
                attention_masks=attention_masks,
                token_type_ids = token_type_ids,
                images=images,
                is_all=self.is_all,
            )
            
            attention_masks = attention_masks.view(-1)
            nonzero_indices = attention_masks.nonzero(as_tuple=True)

            all_logits = all_logits.view(-1, all_logits.size(-1))
            all_labels = all_labels.view(-1)
            filtered_all_logits = all_logits[nonzero_indices]
            filtered_all_probs = torch.softmax(filtered_all_logits, dim=-1)
            filtered_all_labels = all_labels[nonzero_indices]

            loss = self.valid_all_loss(filtered_all_logits, filtered_all_labels)

            self.all_metric.update(filtered_all_probs, filtered_all_labels, is_train=False)
            self.total_metric.update(filtered_all_probs, filtered_all_labels, is_train=False)

            self.valid_all_loss_metric.update(loss)
            self.valid_total_loss_metric.update(loss)
            self.log(
                "train_loss", 
                loss, 
                prog_bar=True,  # 显示在进度条中
                logger=True,    # 记录到TensorBoard
                on_step=True,   # 每step更新
                on_epoch=True,  # 自动计算epoch平均
            )
            return {
                "loss": loss,
                "all_loss": loss,
            }
        else:
            self.is_all = False

            obj_labels = batch["obj_labels"]
            att_labels = batch["att_labels"]
            rel_labels = batch["rel_labels"]
            sce_labels = batch["sce_labels"]
            oth_labels = batch["oth_labels"]

            obj_logits, att_logits, rel_logits, sce_logits, oth_logits = self.model(
                input_ids = input_ids,
                embeddings = embeddings,
                attention_masks=attention_masks,
                token_type_ids = token_type_ids,
                images=images,
                is_all=self.is_all,
            )
            
            attention_masks = attention_masks.view(-1)
            nonzero_indices = attention_masks.nonzero(as_tuple=True)

            obj_logits = obj_logits.view(-1, obj_logits.size(-1))
            obj_labels = obj_labels.view(-1)
            filtered_obj_logits = obj_logits[nonzero_indices]
            filtered_obj_probs = torch.softmax(filtered_obj_logits, dim=-1)
            filtered_obj_labels = obj_labels[nonzero_indices]
            obj_loss = self.valid_obj_loss(filtered_obj_logits, filtered_obj_labels)

            att_logits = att_logits.view(-1, att_logits.size(-1))
            att_labels = att_labels.view(-1)
            filtered_att_logits = att_logits[nonzero_indices]
            filtered_att_probs = torch.softmax(filtered_att_logits, dim=-1)
            filtered_att_labels = att_labels[nonzero_indices]
            att_loss = self.valid_att_loss(filtered_att_logits, filtered_att_labels)

            rel_logits = rel_logits.view(-1, rel_logits.size(-1))
            rel_labels = rel_labels.view(-1)
            filtered_rel_logits = rel_logits[nonzero_indices]
            filtered_rel_probs = torch.softmax(filtered_rel_logits, dim=-1)
            filtered_rel_labels = rel_labels[nonzero_indices]
            rel_loss = self.valid_rel_loss(filtered_rel_logits, filtered_rel_labels)

            sce_logits = sce_logits.view(-1, sce_logits.size(-1))
            sce_labels = sce_labels.view(-1)
            filtered_sce_logits = sce_logits[nonzero_indices]
            filtered_sce_probs = torch.softmax(filtered_sce_logits, dim=-1)
            filtered_sce_labels = sce_labels[nonzero_indices]
            sce_loss = self.valid_sce_loss(filtered_sce_logits, filtered_sce_labels)

            oth_logits = oth_logits.view(-1, oth_logits.size(-1))
            oth_labels = oth_labels.view(-1)
            filtered_oth_logits = oth_logits[nonzero_indices]
            filtered_oth_probs = torch.softmax(filtered_oth_logits, dim=-1)
            filtered_oth_labels = oth_labels[nonzero_indices]
            oth_loss = self.valid_oth_loss(filtered_oth_logits, filtered_oth_labels)

            total_probs = torch.max(filtered_obj_probs[:, 1], filtered_att_probs[:, 1])
            total_probs = torch.max(total_probs, filtered_rel_probs[:, 1])
            total_probs = torch.max(total_probs, filtered_sce_probs[:, 1])
            total_probs = torch.max(total_probs, filtered_oth_probs[:, 1])
            total_probs = torch.stack([1 - total_probs, total_probs], dim=-1)
            total_labels = (filtered_obj_labels | filtered_att_labels | filtered_rel_labels | filtered_sce_labels | filtered_oth_labels)

            loss = obj_loss + att_loss + rel_loss + sce_loss + oth_loss

            # logger.info(f"valid loss: {loss}")
            self.log(
                "valid_loss", 
                loss, 
                prog_bar=True,  # 显示在进度条中
                logger=True,    # 记录到TensorBoard
                on_step=True,   # 每step更新
                on_epoch=True  # 自动计算epoch平均
            )

            self.obj_metric.update(filtered_obj_probs, filtered_obj_labels, is_train=False)
            self.att_metric.update(filtered_att_probs, filtered_att_labels, is_train=False)
            self.rel_metric.update(filtered_rel_probs, filtered_rel_labels, is_train=False)
            self.sce_metric.update(filtered_sce_probs, filtered_sce_labels, is_train=False)
            self.oth_metric.update(filtered_oth_probs, filtered_oth_labels, is_train=False)
            self.total_metric.update(total_probs, total_labels, is_train=False)

            self.valid_obj_loss_metric.update(loss)
            self.valid_att_loss_metric.update(loss)
            self.valid_rel_loss_metric.update(loss)
            self.valid_sce_loss_metric.update(loss)
            self.valid_oth_loss_metric.update(loss)
            self.valid_total_loss_metric.update(loss)

            return {
                "loss": loss,
                "obj_loss": obj_loss,
                "att_loss": att_loss,
                "rel_loss": rel_loss,
                "sce_loss": sce_loss,
                "oth_loss": oth_loss,
            }

    def on_validation_epoch_end(self) -> None:
        loss = self.valid_total_loss_metric.compute()
        self.valid_total_loss_metric.reset()
        
        metric_msg = []
        self.log("valid_loss", loss, prog_bar=True, logger=False, sync_dist=True)
        self.logger.log_metrics(
            metrics={
                "valid_loss": loss,
            },
            step=self.current_epoch,
        )
        metric_msg.append(f"valid_loss: {loss:.4f}")

        if self.is_all:
            for k, v in self.all_metric.compute(is_train=False).items():
                k = f"all/{k}"
                self.log(k, v, prog_bar=False, logger=False, sync_dist=True)
                self.logger.log_metrics(
                    metrics={
                        k: v,
                    },
                    step=self.current_epoch,
                )
                metric_msg.append(f"{k}: {v:.4f}")
        else:
            for k, v in self.obj_metric.compute(is_train=False).items():
                k = f"obj/{k}"
                self.log(k, v, prog_bar=False, logger=False, sync_dist=True)
                self.logger.log_metrics(
                    metrics={
                        k: v,
                    },
                    step=self.current_epoch,
                )
                metric_msg.append(f"{k}: {v:.4f}")

            for k, v in self.att_metric.compute(is_train=False).items():
                k = f"att/{k}"
                self.log(k, v, prog_bar=False, logger=False, sync_dist=True)
                self.logger.log_metrics(
                    metrics={
                        k: v,
                    },
                    step=self.current_epoch,
                )
                metric_msg.append(f"{k}: {v:.4f}")

            for k, v in self.rel_metric.compute(is_train=False).items():
                k = f"rel/{k}"
                self.log(k, v, prog_bar=False, logger=False, sync_dist=True)
                self.logger.log_metrics(
                    metrics={
                        k: v,
                    },
                    step=self.current_epoch,
                )
                metric_msg.append(f"{k}: {v:.4f}")

            for k, v in self.sce_metric.compute(is_train=False).items():
                k = f"sce/{k}"
                self.log(k, v, prog_bar=False, logger=False, sync_dist=True)
                self.logger.log_metrics(
                    metrics={
                        k: v,
                    },
                    step=self.current_epoch,
                )
                metric_msg.append(f"{k}: {v:.4f}")

            for k, v in self.oth_metric.compute(is_train=False).items():
                k = f"oth/{k}"
                self.log(k, v, prog_bar=False, logger=False, sync_dist=True)
                self.logger.log_metrics(
                    metrics={
                        k: v,
                    },
                    step=self.current_epoch,
                )
                metric_msg.append(f"{k}: {v:.4f}")

            for k, v in self.total_metric.compute(is_train=False).items():
                k = f"total/{k}"
                self.log(k, v, prog_bar=False, logger=False, sync_dist=True)
                self.logger.log_metrics(
                    metrics={
                        k: v,
                    },
                    step=self.current_epoch,
                )
                metric_msg.append(f"{k}: {v:.4f}")

        metric_msg = "\n".join(metric_msg)
        msg = (
            f"[Epoch: {self.current_epoch + 1}/{self.trainer.max_epochs}] [Valid Evaluation]\n"
            f"{metric_msg}"
        )
        self.custom_logger.info(msg)