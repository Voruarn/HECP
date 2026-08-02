# import os
# os.environ['OMP_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.datamodule import register_datamodule
from src.datamodule.base import BaseDataModule
from src.message import SupervisionDataModuleMessage
import torch.distributed as dist
from torch.utils.data import RandomSampler


def pad_high_dim_tensors(tensors, seq_dim, pad_value=0):
    """
    对高维张量进行 Padding。
    支持将序列维度（seq_len）不在第 0 维的张量进行对齐，并自动处理 Attention 的方阵双维度。
    
    Args:
        tensors: List[torch.Tensor], 批次中的张量列表
        seq_dim: int, 序列长度所在的维度索引 (例如 hidden_states 为 1, attentions 为 2)
        pad_value: 填充值，默认为 0
    """
    if not tensors:
        return None
    
    batch_size = len(tensors)
    max_seq_len = max(t.shape[seq_dim] for t in tensors)
    
    # 构建目标形状 (加上 batch 维度)
    target_shape = list(tensors[0].shape)
    target_shape[seq_dim] = max_seq_len
    
    # 启发式检测是否为 Attention 方阵 (例如 shape: [L, H, S, S]，seq_dim=2)
    is_square_matrix = False
    if seq_dim + 1 < len(target_shape) and tensors[0].shape[seq_dim] == tensors[0].shape[seq_dim + 1]:
        is_square_matrix = True
        target_shape[seq_dim + 1] = max_seq_len

    # 创建填充后的目标张量
    padded = torch.full(
        [batch_size] + target_shape, 
        pad_value, 
        dtype=tensors[0].dtype, 
        device=tensors[0].device
    )
    
    # 逐个样本填充数据
    for i, t in enumerate(tensors):
        seq_len = t.shape[seq_dim]
        
        # 构建动态切片索引
        slices = [i] + [slice(None)] * len(target_shape)
        slices[seq_dim + 1] = slice(0, seq_len) # +1 是因为第 0 维是 batch
        
        if is_square_matrix:
            slices[seq_dim + 2] = slice(0, seq_len)
            
        padded[tuple(slices)] = t
        
    return padded


@register_datamodule(name="hecp")
class HallocDataModule(BaseDataModule):
    def get_datamodule_msg(self) -> SupervisionDataModuleMessage:
        num_total_devices = self.hparams.trainer_msg.num_nodes * self.hparams.trainer_msg.num_devices
        max_steps = len(self.train_dataset) // (self.hparams.batch_size_per_gpu * num_total_devices) \
            * self.hparams.trainer_msg.max_epochs
        return SupervisionDataModuleMessage(
            classes=self.train_dataset.classes,
            max_steps=max_steps,
        )
    
    def collate_fn(self, batch):
        # 1. 提取基础特征
        input_ids = [item["input_ids"] for item in batch]
        token_type_ids = [item["token_type_ids"] for item in batch]
        attention_masks = [torch.tensor(item["attention_masks"]) for item in batch]
        image_paths = [item["image_paths"] for item in batch]
        
        # 对 attention_masks 使用标准 pad_sequence (序列在第 0 维)
        attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
        input_ids = [torch.tensor(input_id) for input_id in input_ids]
        token_type_ids = [torch.tensor(token_type_id) for token_type_id in token_type_ids]
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
        token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
        
        result_dict = {
            "attention_masks": attention_masks,
            "image_paths": image_paths,
            "input_ids": input_ids,
            "token_type_ids": token_type_ids,
        }

        # 2. 提取并 Padding 新增的高维特征
        # hidden_states shape: [num_layers, seq_len, hidden_dim] -> seq_dim = 1
        hidden_states = [torch.tensor(item["hidden_states"]) for item in batch]
        result_dict["hidden_states"] = pad_high_dim_tensors(hidden_states, seq_dim=1, pad_value=0)
            
        # 3. 提取并 Padding 标签
        if "all_labels" in batch[0]:
            all_labels = [torch.tensor(item["all_labels"]) for item in batch]
            result_dict["all_labels"] = pad_sequence(all_labels, batch_first=True, padding_value=0)
        else:
            label_keys = ["obj_labels", "att_labels", "rel_labels", "sce_labels", "oth_labels"]
            for key in label_keys:
                labels = [torch.tensor(item[key]) for item in batch]
                result_dict[key] = pad_sequence(labels, batch_first=True, padding_value=0)

        return result_dict
    
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(self.train_dataset, shuffle=True)
        else:
            sampler = RandomSampler(self.train_dataset)
            
        return DataLoader(self.train_dataset,
                            batch_size=self.hparams.batch_size_per_gpu,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            sampler=sampler,
                            collate_fn=self.collate_fn,
                            prefetch_factor=2)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(self.valid_dataset, shuffle=False)
        else:
            sampler = RandomSampler(self.valid_dataset)
            
        return DataLoader(self.valid_dataset,
                          batch_size=self.hparams.batch_size_per_gpu,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          sampler=sampler,
                          collate_fn=self.collate_fn,
                          prefetch_factor=2)
    
    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(self.test_dataset, shuffle=False)
        else:
            sampler = RandomSampler(self.test_dataset)
               
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size_per_gpu,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          sampler=sampler,
                          collate_fn=self.collate_fn,
                          prefetch_factor=2)