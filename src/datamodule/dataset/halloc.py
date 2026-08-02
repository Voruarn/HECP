import os
from typing import Dict

import numpy as np
from tqdm import tqdm

from src.datamodule.dataset import register_dataset
from src.datamodule.dataset.base import BaseDataset


@register_dataset(name="halloc")
class HallocDataset(BaseDataset):
    def __init__(
        self,
        file_paths: str,
        all_flag: bool,
    ) -> None:
        self.all_flag = all_flag
        super().__init__(file_paths=file_paths)

    def _read_single_file(self, file_path: str) -> Dict:
        embeddings = np.load(os.path.join(file_path, "embeddings.npy"), mmap_mode='r')
        attention_masks = np.load(os.path.join(file_path, "attention_masks.npy"), mmap_mode='r')
        image_paths = np.load(os.path.join(file_path, "image_paths.npy"), mmap_mode='r')

        all_label_path = os.path.join(file_path, "all_labels.npy")
        if self.all_flag and os.path.exists(all_label_path):
            all_labels = np.load(all_label_path, mmap_mode='r')
            return embeddings, attention_masks, image_paths, all_labels
        else:
            obj_labels = np.load(os.path.join(file_path, "obj_labels.npy"), mmap_mode='r')
            att_labels = np.load(os.path.join(file_path, "att_labels.npy"), mmap_mode='r')
            rel_labels = np.load(os.path.join(file_path, "rel_labels.npy"), mmap_mode='r')
            sce_labels = np.load(os.path.join(file_path, "sce_labels.npy"), mmap_mode='r')
            oth_labels = np.load(os.path.join(file_path, "oth_labels.npy"), mmap_mode='r')
            return embeddings, attention_masks, image_paths, obj_labels, att_labels, rel_labels, sce_labels, oth_labels

    def _prepare(self) -> None:
        self.embeddings_bag = []
        self.attention_masks_bag = []
        self.image_paths_bag = []
        self.all_labels_bag = []
        self.obj_labels_bag = []
        self.att_labels_bag = []
        self.rel_labels_bag = []
        self.sce_labels_bag = []
        self.oth_labels_bag = []

        file_paths = [p.strip() for p in self.file_paths.split(",")]
        for file_path in tqdm(file_paths):
            if self.all_flag:
                embeddings, attention_masks, image_paths, all_labels = self._read_single_file(file_path)
                
                # check nan
                if np.isnan(embeddings).any():
                    print(f"Found nan in {file_path} ({np.isnan(embeddings).sum()} nans)")
                    continue

                self.all_labels_bag.append(all_labels)
            else:
                embeddings, attention_masks, image_paths, obj_labels, att_labels, rel_labels, sce_labels, oth_labels = self._read_single_file(file_path)

                # check nan
                if np.isnan(embeddings).any():
                    print(f"Found nan in {file_path} ({np.isnan(embeddings).sum()} nans)")
                    continue

                self.obj_labels_bag.append(obj_labels)
                self.att_labels_bag.append(att_labels)
                self.rel_labels_bag.append(rel_labels)
                self.sce_labels_bag.append(sce_labels)
                self.oth_labels_bag.append(oth_labels)
            self.embeddings_bag.append(embeddings)
            self.attention_masks_bag.append(attention_masks)
            self.image_paths_bag.append(image_paths)

        self.embeddings_bag_map = {}
        idx = 0
        for bag_idx, bag in enumerate(self.embeddings_bag):
            for new_idx in range(len(bag)):
                self.embeddings_bag_map[idx] = (bag_idx, new_idx)
                idx += 1

    def __getitem__(self, idx: int) -> Dict:
        bag_idx, new_idx = self.embeddings_bag_map[idx]

        if self.all_flag:
            return {
                "embeddings": self.embeddings_bag[bag_idx][new_idx],
                "attention_masks": self.attention_masks_bag[bag_idx][new_idx],
                "image_paths": self.image_paths_bag[bag_idx][new_idx],
                "all_labels": self.all_labels_bag[bag_idx][new_idx],
            }
        else:
            return {
                "embeddings": self.embeddings_bag[bag_idx][new_idx],
                "attention_masks": self.attention_masks_bag[bag_idx][new_idx],
                "image_paths": self.image_paths_bag[bag_idx][new_idx],
                "obj_labels": self.obj_labels_bag[bag_idx][new_idx],
                "att_labels": self.att_labels_bag[bag_idx][new_idx],
                "rel_labels": self.rel_labels_bag[bag_idx][new_idx],
                "sce_labels": self.sce_labels_bag[bag_idx][new_idx],
                "oth_labels": self.oth_labels_bag[bag_idx][new_idx],
            }
    
    @property
    def classes(self) -> list:
        return [0, 1]
    
    def __len__(self) -> int:
        return sum([len(bag) for bag in self.embeddings_bag])
