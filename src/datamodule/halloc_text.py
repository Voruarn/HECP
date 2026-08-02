import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from src.datamodule import register_datamodule
from src.datamodule.base import BaseDataModule
from src.message import SupervisionDataModuleMessage


@register_datamodule(name="halloc_text")
class HallocTextDataModule(BaseDataModule):
    def get_datamodule_msg(self) -> SupervisionDataModuleMessage:
        num_total_devices = self.hparams.trainer_msg.num_nodes * self.hparams.trainer_msg.num_devices
        max_steps = len(self.train_dataset) // (self.hparams.batch_size_per_gpu * num_total_devices) \
            * self.hparams.trainer_msg.max_epochs
        return SupervisionDataModuleMessage(
            classes=self.train_dataset.classes,
            max_steps=max_steps,
        )
    
    def collate_fn(self, batch):
        if "all_labels" in batch[0].keys():
            input_ids = [item["input_ids"] for item in batch]
            attention_masks = [item["attention_masks"] for item in batch]
            token_type_ids = [item["token_type_ids"] for item in batch]
            image_paths = [item["image_paths"] for item in batch]
            all_labels = [item["all_labels"] for item in batch]

            input_ids = [torch.tensor(input_id) for input_id in input_ids]
            attention_masks = [torch.tensor(attention_mask) for attention_mask in attention_masks]
            token_type_ids = [torch.tensor(token_type_id) for token_type_id in token_type_ids]
            all_labels = [torch.tensor(labels) for labels in all_labels]

            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
            all_labels = pad_sequence(all_labels, batch_first=True, padding_value=0)

            return {
                "input_ids": input_ids,
                "attention_masks": attention_masks,
                "token_type_ids": token_type_ids,
                "image_paths": image_paths,
                "all_labels": all_labels,
            }
        else:
            input_ids = [item["input_ids"] for item in batch]
            attention_masks = [item["attention_masks"] for item in batch]
            token_type_ids = [item["token_type_ids"] for item in batch]
            image_paths = [item["image_paths"] for item in batch]
            obj_labels = [item["obj_labels"] for item in batch]
            att_labels = [item["att_labels"] for item in batch]
            rel_labels = [item["rel_labels"] for item in batch]
            sce_labels = [item["sce_labels"] for item in batch]
            oth_labels = [item["oth_labels"] for item in batch]
            all_labels = [item["all_labels"] for item in batch]

            input_ids = [torch.tensor(input_id) for input_id in input_ids]
            attention_masks = [torch.tensor(attention_mask) for attention_mask in attention_masks]
            token_type_ids = [torch.tensor(token_type_id) for token_type_id in token_type_ids]
            obj_labels = [torch.tensor(labels) for labels in obj_labels]
            att_labels = [torch.tensor(labels) for labels in att_labels]
            rel_labels = [torch.tensor(labels) for labels in rel_labels]
            sce_labels = [torch.tensor(labels) for labels in sce_labels]
            oth_labels = [torch.tensor(labels) for labels in oth_labels]
            all_labels = [torch.tensor(labels) for labels in all_labels]

            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
            attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
            token_type_ids = pad_sequence(token_type_ids, batch_first=True, padding_value=0)
            obj_labels = pad_sequence(obj_labels, batch_first=True, padding_value=0)
            att_labels = pad_sequence(att_labels, batch_first=True, padding_value=0)
            rel_labels = pad_sequence(rel_labels, batch_first=True, padding_value=0)
            sce_labels = pad_sequence(sce_labels, batch_first=True, padding_value=0)
            oth_labels = pad_sequence(oth_labels, batch_first=True, padding_value=0)
            all_labels = pad_sequence(all_labels, batch_first=True, padding_value=0)

            return {
                "input_ids": input_ids,
                "attention_masks": attention_masks,
                "token_type_ids": token_type_ids,
                "image_paths": image_paths,
                "obj_labels": obj_labels,
                "att_labels": att_labels,
                "rel_labels": rel_labels,
                "sce_labels": sce_labels,
                "oth_labels": oth_labels,
                "all_labels": all_labels
            }
    
    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        sampler = DistributedSampler(self.train_dataset, shuffle=True)
        return DataLoader(self.train_dataset,
                            batch_size=self.hparams.batch_size_per_gpu,
                            num_workers=self.hparams.num_workers,
                            pin_memory=self.hparams.pin_memory,
                            sampler=sampler,
                            collate_fn=self.collate_fn)

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        sampler = DistributedSampler(self.valid_dataset, shuffle=False)
        return DataLoader(self.valid_dataset,
                          batch_size=self.hparams.batch_size_per_gpu,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          sampler=sampler,
                          collate_fn=self.collate_fn)
    
    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        sampler = DistributedSampler(self.test_dataset, shuffle=False)
        return DataLoader(self.test_dataset,
                          batch_size=self.hparams.batch_size_per_gpu,
                          num_workers=self.hparams.num_workers,
                          pin_memory=self.hparams.pin_memory,
                          sampler=sampler,
                          collate_fn=self.collate_fn)
