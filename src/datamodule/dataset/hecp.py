import os
from typing import Dict

import numpy as np
from tqdm import tqdm

from src.datamodule.dataset import register_dataset
from src.datamodule.dataset.base import BaseDataset


@register_dataset(name="hecp")
class HallocDataset(BaseDataset):
    def __init__(
        self,
        file_paths: str,
        all_flag: bool,
    ) -> None:
        self.all_flag = all_flag
        super().__init__(file_paths=file_paths)

    def _read_single_file(self, file_path: str) -> Dict:
        # 1. 读取原有的 .npy 文件      
        input_ids = np.load(os.path.join(file_path, "input_ids.npy"), mmap_mode='r')
        token_type_ids = np.load(os.path.join(file_path, "token_type_ids.npy"), mmap_mode='r')
       
        attention_masks = np.load(os.path.join(file_path, "attention_masks.npy"), mmap_mode='r')
        image_paths = np.load(os.path.join(file_path, "image_paths.npy"), mmap_mode='r')

        # 2. 读取 hidden_states 从 .npy 文件
        npy_hidden_states_path = os.path.join(file_path, "hidden_states_selected_layers.npy")
        # 使用 mmap_mode='r' 进行内存映射读取，避免大文件 OOM，且完美支持多进程 DataLoader
        hidden_states = np.load(npy_hidden_states_path, mmap_mode='r') if os.path.exists(npy_hidden_states_path) else None
        
        # 3. 读取 labels
        all_label_path = os.path.join(file_path, "all_labels.npy")
        if self.all_flag and os.path.exists(all_label_path):
            all_labels = np.load(all_label_path, mmap_mode='r')
            return (attention_masks, image_paths, all_labels, 
                    hidden_states, input_ids, token_type_ids)
        else:
            obj_labels = np.load(os.path.join(file_path, "obj_labels.npy"), mmap_mode='r')
            att_labels = np.load(os.path.join(file_path, "att_labels.npy"), mmap_mode='r')
            rel_labels = np.load(os.path.join(file_path, "rel_labels.npy"), mmap_mode='r')
            sce_labels = np.load(os.path.join(file_path, "sce_labels.npy"), mmap_mode='r')
            oth_labels = np.load(os.path.join(file_path, "oth_labels.npy"), mmap_mode='r')
            return (attention_masks, image_paths, 
                    obj_labels, att_labels, rel_labels, sce_labels, oth_labels,
                    hidden_states, input_ids, token_type_ids)

    def _prepare(self) -> None:
        # 原有 bags
        self.attention_masks_bag = []
        self.image_paths_bag = []
        self.all_labels_bag = []
        self.obj_labels_bag = []
        self.att_labels_bag = []
        self.rel_labels_bag = []
        self.sce_labels_bag = []
        self.oth_labels_bag = []
        
        # 新增 bags
        self.hidden_states_bag = []
        self.input_ids_bag = []
        self.token_type_ids_bag = []

        file_paths = [p.strip() for p in self.file_paths.split(",")]
        for file_path in tqdm(file_paths):
            if self.all_flag:
                (attention_masks, image_paths, all_labels, 
                 hidden_states, input_ids, token_type_ids) = self._read_single_file(file_path)
                
                # check nan
                if np.isnan(attention_masks).any():
                    print(f"Found nan in {file_path} ({np.isnan(attention_masks).sum()} nans)")
                    continue

                self.all_labels_bag.append(all_labels)
            else:
                (attention_masks, image_paths, 
                 obj_labels, att_labels, rel_labels, sce_labels, oth_labels,
                 hidden_states, input_ids, token_type_ids) = self._read_single_file(file_path)

                if np.isnan(attention_masks).any():
                    print(f"Found nan in {file_path} ({np.isnan(attention_masks).sum()} nans)")
                    continue

                self.obj_labels_bag.append(obj_labels)
                self.att_labels_bag.append(att_labels)
                self.rel_labels_bag.append(rel_labels)
                self.sce_labels_bag.append(sce_labels)
                self.oth_labels_bag.append(oth_labels)
                
            self.attention_masks_bag.append(attention_masks)
            self.image_paths_bag.append(image_paths)
            
            # 追加新特征
            self.hidden_states_bag.append(hidden_states)
            self.input_ids_bag.append(input_ids)
            self.token_type_ids_bag.append(token_type_ids)

        # 构建全局索引映射
        self.bag_map = {}
        idx = 0
        for bag_idx, bag in enumerate(self.attention_masks_bag):
            for new_idx in range(len(bag)):
                self.bag_map[idx] = (bag_idx, new_idx)
                idx += 1

    def __getitem__(self, idx: int) -> Dict:
        bag_idx, new_idx = self.bag_map[idx]

        base_dict = {
            "attention_masks": self.attention_masks_bag[bag_idx][new_idx],
            "image_paths": self.image_paths_bag[bag_idx][new_idx],
            "hidden_states": self.hidden_states_bag[bag_idx][new_idx],
            "input_ids": self.input_ids_bag[bag_idx][new_idx],
            "token_type_ids": self.token_type_ids_bag[bag_idx][new_idx],
        }

        if self.all_flag:
            base_dict["all_labels"] = self.all_labels_bag[bag_idx][new_idx]
        else:
            base_dict.update({
                "obj_labels": self.obj_labels_bag[bag_idx][new_idx],
                "att_labels": self.att_labels_bag[bag_idx][new_idx],
                "rel_labels": self.rel_labels_bag[bag_idx][new_idx],
                "sce_labels": self.sce_labels_bag[bag_idx][new_idx],
                "oth_labels": self.oth_labels_bag[bag_idx][new_idx],
            })

        return base_dict
    
    @property
    def classes(self) -> list:
        return [0, 1]
    
    def __len__(self) -> int:
        return sum([len(bag) for bag in self.attention_masks_bag])