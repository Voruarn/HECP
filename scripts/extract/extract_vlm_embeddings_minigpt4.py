import os
import sys
current_working_dir = os.getcwd()
sys.path.append(current_working_dir + '/MiniGPT-4')

import gc
import json
from typing import Optional

import fire
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from omegaconf import DictConfig
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from minigpt4.common.config import Config
from minigpt4.common.registry import registry
from minigpt4.processors.blip_processors import BlipCaptionProcessor

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'


def read_image(image_id: str, image_dir: str, is_vg: bool) -> Image.Image:
    if is_vg: # VISUAL GENOME
        primary_path = os.path.join(image_dir, "VG_100K", f"{image_id}.jpg")
        secondary_path = os.path.join(image_dir, "VG_100K_2", f"{image_id}.jpg")
        if os.path.exists(primary_path):
            return Image.open(primary_path).convert("RGB"), primary_path
        elif os.path.exists(secondary_path):
            return Image.open(secondary_path).convert("RGB"), secondary_path
        else:
            raise FileNotFoundError(f"Image with ID {image_id} not found in {image_dir}")
    else: # COCO
        primary_path = os.path.join(image_dir, "train2014", f"{image_id}")
        secondary_path = os.path.join(image_dir, "val2014", f"{image_id}")
        if os.path.exists(primary_path):
            return Image.open(primary_path).convert("RGB"), primary_path
        elif os.path.exists(secondary_path):
            return Image.open(secondary_path).convert("RGB"), secondary_path
        else:
            raise FileNotFoundError(f"Image with ID not found in {primary_path}, {secondary_path}")
    

def load_model(gpu_id):
    cfg = Config(DictConfig({
        "options": None,
        "cfg_path": "MiniGPT-4/eval_configs/minigpt4_eval.yaml"
    }))
    model_config = cfg.model_cfg
    model_config.device_8bit = gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config)
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
    text_processor = BlipCaptionProcessor()
    return model, vis_processor, text_processor


def load_dataset(data_path: str, image_dir: str, is_vg: bool):
    with open(data_path, "r") as f:
        data = json.load(f)

    dataset = []
    for idx, item in enumerate(data):
        data_id = item["id"]
        image_id = item["image_id"]
        prompt = item["prompt"]
        text = item["hallucinated_text"]
        annotations = item["annotations"]
        tokenized_text = item["tokenized_text"]

        prompt = prompt.replace("<image>", "").strip()
        
        image, image_path = read_image(image_id, image_dir, is_vg)

        obj_h_token_indices = []
        att_h_token_indices = []
        rel_h_token_indices = []
        sce_h_token_indices = []
        oth_h_token_indices = []
        all_h_token_indices = []
        for h_type, h_list in annotations.items():
            for h in h_list:
                for k in h.keys():
                    hallucinated_answer = h[k]
                    token_index_splits = hallucinated_answer["token_index"].split(":")

                    if len(token_index_splits) == 1:
                        new_token_indices = [int(token_index_splits[0])]
                    else:
                        start, end = map(int, token_index_splits)
                        new_token_indices = list(range(start, end))

                    if h_type == "object":
                        obj_h_token_indices.extend(new_token_indices)
                        all_h_token_indices.extend(new_token_indices)
                    elif h_type == "attribute":
                        att_h_token_indices.extend(new_token_indices)
                        all_h_token_indices.extend(new_token_indices)
                    elif h_type == "relationship":
                        rel_h_token_indices.extend(new_token_indices)
                        all_h_token_indices.extend(new_token_indices)
                    elif h_type == "scene":
                        sce_h_token_indices.extend(new_token_indices)
                        all_h_token_indices.extend(new_token_indices)
                    elif h_type == "other":
                        oth_h_token_indices.extend(new_token_indices)
                        all_h_token_indices.extend(new_token_indices)
                    else:
                        if h_type == "all":
                            all_h_token_indices.extend(new_token_indices)
                        else:
                            raise ValueError(f"Unknown hallucination type: {h_type}")

        token_length = len(tokenized_text)
        obj_label = torch.full((token_length,), 0, dtype=torch.long)
        att_label = torch.full((token_length,), 0, dtype=torch.long)
        rel_label = torch.full((token_length,), 0, dtype=torch.long)
        sce_label = torch.full((token_length,), 0, dtype=torch.long)
        oth_label = torch.full((token_length,), 0, dtype=torch.long)
        all_label = torch.full((token_length,), 0, dtype=torch.long)
        label_mask = torch.full((token_length,), 1, dtype=torch.long)

        for i in range(token_length):
            if i in obj_h_token_indices:
                obj_label[i] = 1
            if i in att_h_token_indices:
                att_label[i] = 1
            if i in rel_h_token_indices:
                rel_label[i] = 1
            if i in sce_h_token_indices:
                sce_label[i] = 1
            if i in oth_h_token_indices:
                oth_label[i] = 1
            if i in all_h_token_indices:
                all_label[i] = 1

        dataset.append({
            "index": idx,
            "id": data_id,
            "image": image,
            "image_path": image_path,
            "prompt": prompt,
            "text": text,
            "obj_label": obj_label,
            "att_label": att_label,
            "rel_label": rel_label,
            "sce_label": sce_label,
            "oth_label": oth_label,
            "all_label": all_label,
            "label_mask": label_mask,
        })

    return dataset


def pad_tensor(tensor, target_shape):
    if len(tensor.shape) != len(target_shape):
        raise ValueError("Tensor and target shape must have the same number of dimensions")
    
    padding = []
    for i in range(len(tensor.shape)-1, 0, -1):
        if tensor.shape[i] < target_shape[i]:
            padding.extend([0, target_shape[i] - tensor.shape[i]])
        else:
            padding.extend([0, 0])
    
    padding.extend([0, 0])  # Ensure no padding for the 0th dimension
    return F.pad(tensor, padding, mode='constant', value=0)


def main(
    target_checkpoint: Optional[str] = None,
    save_dir: str = "data/mhaldetect/minigpt4/val",
    data_path: str = "json_data/mhaldetect_val_haloc_char_index_minigpt4_postprocessed.json",
    image_dir: str = "",
    batch_size: int = 16,
    num_workers: int = 8,
    is_vg: bool = True,
):
    os.makedirs(save_dir, exist_ok=True)

    accelerator = Accelerator()

    num_gpus = torch.cuda.device_count()
    accelerator.print(f"Number of GPUs available: {num_gpus}")
    accelerator.print(f"Accelerator distributed: {accelerator.use_distributed}")

    accelerator.print("[args] target_checkpoint:", target_checkpoint)
    accelerator.print("[args] save_dir:", save_dir)
    accelerator.print("[args] data_path:", data_path)
    accelerator.print("[args] image_dir:", image_dir)
    accelerator.print("[args] batch_size:", batch_size)
    accelerator.print("[args] num_workers:", num_workers)

    accelerator.print("Load model")

    # TODO: check if the config is correct
    model, vis_processor, text_processor = load_model(gpu_id=accelerator.device.index)

    accelerator.print("Load tartget checkpoint")
    if target_checkpoint is not None:
        model.load_state_dict(torch.load(target_checkpoint, map_location=accelerator.device), strict=False)

    accelerator.print("Define collate fn")
    def collate_fn(batch):
        index = [item["index"] for item in batch]
        id = [item["id"] for item in batch]
        image = torch.stack([vis_processor(item["image"]) for item in batch])
        image_path = [item["image_path"] for item in batch]
        prompt = ["<Img><ImageHere></Img> {} ".format(item["prompt"]) for item in batch]  # TODO: check if this is correct
        # prompt = [text_processor(item["prompt"]) for item in batch]  # TODO: check if this is correct
        text = [text_processor(item["text"]) for item in batch]
        obj_label = pad_sequence([item["obj_label"] for item in batch], batch_first=True, padding_value=0)
        att_label = pad_sequence([item["att_label"] for item in batch], batch_first=True, padding_value=0)
        rel_label = pad_sequence([item["rel_label"] for item in batch], batch_first=True, padding_value=0)
        sce_label = pad_sequence([item["sce_label"] for item in batch], batch_first=True, padding_value=0)
        oth_label = pad_sequence([item["oth_label"] for item in batch], batch_first=True, padding_value=0)
        all_label = pad_sequence([item["all_label"] for item in batch], batch_first=True, padding_value=0)
        label_mask = pad_sequence([item["label_mask"] for item in batch], batch_first=True, padding_value=0)
        return {
            "index": index,
            "id": id,
            "image": image,
            "image_path": image_path,
            "prompt": prompt,
            "text": text,
            "obj_label": obj_label,
            "att_label": att_label,
            "rel_label": rel_label,
            "sce_label": sce_label,
            "oth_label": oth_label,
            "all_label": all_label,
            "label_mask": label_mask,
        }

    accelerator.print("Load dataset")
    test_dataset = load_dataset(data_path, image_dir, is_vg)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )
    accelerator.print(f"Total test dataset size: {len(test_dataset)}")

    accelerator.print("Prepare model and dataloader")
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    accelerator.print("Start inference")

    indices_results = []
    ids_results = []
    image_paths_results = []

    embeddings_results = None
    attention_masks_results = None
    obj_labels_results = None
    att_labels_results = None
    rel_labels_results = None
    sce_labels_results = None
    oth_labels_results = None
    all_labels_results = None
    label_masks_results = None

    try:
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                index = batch["index"]
                id = batch["id"]
                image = batch["image"].to(accelerator.device)
                image_path = batch["image_path"]
                text_input = batch["prompt"]
                text_output = batch["text"]
                obj_label = batch["obj_label"].to(accelerator.device)
                att_label = batch["att_label"].to(accelerator.device)
                rel_label = batch["rel_label"].to(accelerator.device)
                sce_label = batch["sce_label"].to(accelerator.device)
                oth_label = batch["oth_label"].to(accelerator.device)
                all_label = batch["all_label"].to(accelerator.device)
                label_mask = batch["label_mask"].to(accelerator.device)

                outputs = model({
                    "image": image,
                    "instruction_input": text_input,
                    "answer": text_output,
                }, return_output_embeddings=True)
                embeddings, attention_masks = outputs

                num_zeros_in_attention_masks = [torch.sum(mask == 0).item() for mask in attention_masks]
                num_ones_in_label_masks = [torch.sum(mask == 1).item() for mask in label_mask]

                new_embeddings = []
                new_attention_masks = []
                for (
                    embedding,
                    attention_mask,
                    num_zeros_in_attention_mask,
                    num_ones_in_label_mask,
                ) in zip(
                    embeddings,
                    attention_masks,
                    num_zeros_in_attention_masks,
                    num_ones_in_label_masks
                ):
                    start = -(num_zeros_in_attention_mask + num_ones_in_label_mask)
                    end = -num_zeros_in_attention_mask
                    if end == 0:
                        end = None
                    new_embedding = embedding[start:end]
                    new_attention_mask = attention_mask[start:end]
                    new_embeddings.append(new_embedding)
                    new_attention_masks.append(new_attention_mask)

                embeddings = pad_sequence(
                    [t for t in new_embeddings],
                    batch_first=True,
                    padding_value=0,
                )
                attention_masks = pad_sequence(
                    [t for t in new_attention_masks],
                    batch_first=True,
                    padding_value=0,
                )

                accelerator.print(f"[intermediate] embeddings shape: {embeddings.shape}")
                accelerator.print(f"[intermediate] attention_masks shape: {attention_masks.shape}")
                accelerator.print(f"[intermediate] obj_labels shape: {obj_label.shape}")
                accelerator.print(f"[intermediate] att_labels shape: {att_label.shape}")
                accelerator.print(f"[intermediate] rel_labels shape: {rel_label.shape}")
                accelerator.print(f"[intermediate] sce_labels shape: {sce_label.shape}")
                accelerator.print(f"[intermediate] oth_labels shape: {oth_label.shape}")
                accelerator.print(f"[intermediate] all_labels shape: {all_label.shape}")
                accelerator.print(f"[intermediate] label_masks shape: {label_mask.shape}")

                embeddings = accelerator.pad_across_processes(
                    embeddings, dim=1
                )

                attention_masks = accelerator.pad_across_processes(
                    attention_masks, dim=1
                )

                obj_label = accelerator.pad_across_processes(
                    obj_label, dim=1
                )

                att_label = accelerator.pad_across_processes(
                    att_label, dim=1
                )

                rel_label = accelerator.pad_across_processes(
                    rel_label, dim=1
                )

                sce_label = accelerator.pad_across_processes(
                    sce_label, dim=1
                )

                oth_label = accelerator.pad_across_processes(
                    oth_label, dim=1
                )

                all_label = accelerator.pad_across_processes(
                    all_label, dim=1
                )

                label_mask = accelerator.pad_across_processes(
                    label_mask, dim=1
                )

                embeddings = accelerator.gather(embeddings).cpu()
                attention_masks = accelerator.gather(attention_masks).cpu()
                obj_label = accelerator.gather(obj_label).cpu()
                att_label = accelerator.gather(att_label).cpu()
                rel_label = accelerator.gather(rel_label).cpu()
                sce_label = accelerator.gather(sce_label).cpu()
                oth_label = accelerator.gather(oth_label).cpu()
                all_label = accelerator.gather(all_label).cpu()
                label_mask = accelerator.gather(label_mask).cpu()

                index = torch.tensor(index, device=accelerator.device)
                index = accelerator.gather(index).cpu().numpy().tolist()

                id = accelerator.gather_for_metrics(id)
                image_path = accelerator.gather_for_metrics(image_path)

                if accelerator.is_main_process:
                    if embeddings_results is None:
                        embeddings_results = embeddings
                        attention_masks_results = attention_masks
                    else:
                        embeddings_results = pad_sequence(
                            [t for t in embeddings_results] + [t for t in embeddings],
                            batch_first=True,
                            padding_value=0,
                        ) # (batch_size, max_seq_len, hidden_size)
                        attention_masks_results = pad_sequence(
                            [t for t in attention_masks_results] + [t for t in attention_masks],
                            batch_first=True,
                            padding_value=0,
                        )

                    if obj_labels_results is None:
                        obj_labels_results = obj_label
                        att_labels_results = att_label
                        rel_labels_results = rel_label
                        sce_labels_results = sce_label
                        oth_labels_results = oth_label
                        all_labels_results = all_label
                        label_masks_results = label_mask
                    else:
                        obj_labels_results = pad_sequence(
                            [t for t in obj_labels_results] + [t for t in obj_label],
                            batch_first=True,
                            padding_value=0,
                        )
                        att_labels_results = pad_sequence(
                            [t for t in att_labels_results] + [t for t in att_label],
                            batch_first=True,
                            padding_value=0,
                        )
                        rel_labels_results = pad_sequence(
                            [t for t in rel_labels_results] + [t for t in rel_label],
                            batch_first=True,
                            padding_value=0,
                        )
                        sce_labels_results = pad_sequence(
                            [t for t in sce_labels_results] + [t for t in sce_label],
                            batch_first=True,
                            padding_value=0,
                        )
                        oth_labels_results = pad_sequence(
                            [t for t in oth_labels_results] + [t for t in oth_label],
                            batch_first=True,
                            padding_value=0,
                        )
                        all_labels_results = pad_sequence(
                            [t for t in all_labels_results] + [t for t in all_label],
                            batch_first=True,
                            padding_value=0,
                        )
                        label_masks_results = pad_sequence(
                            [t for t in label_masks_results] + [t for t in label_mask],
                            batch_first=True,
                            padding_value=0,
                        )
                    indices_results.extend(index)
                    ids_results.extend(id)
                    image_paths_results.extend(image_path)

                gc.collect()
                torch.cuda.empty_cache()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            indices_results = np.array(indices_results)
            unique, unique_positions = np.unique(indices_results, return_index=True)
            # sorted_unique_positions = np.argsort(unique_positions)

            accelerator.print(f"shape of embeddings_results: {embeddings_results.shape}")
            accelerator.print(f"shape of attention_masks_results: {attention_masks_results.shape}")
            accelerator.print(f"shape of unique positions: {unique_positions.shape}")
            # accelerator.print(f"shape of sorted unique positions: {sorted_unique_positions.shape}")
            accelerator.print(f"unique_positions[:10]: {unique_positions[:10]}")
            # accelerator.print(f"sorted_unique_positions[:10]: {sorted_unique_positions[:10]}")
            accelerator.print(f"unique_positions[-10:]: {unique_positions[-10:]}")
            # accelerator.print(f"sorted_unique_positions[-10:]: {sorted_unique_positions[-10:]}")
            # for i, j in zip(unique_positions, sorted_unique_positions):
                # assert i == j

            embeddings_results = embeddings_results[unique_positions]
            attention_masks_results = attention_masks_results[unique_positions]
            obj_labels_results = obj_labels_results[unique_positions]
            att_labels_results = att_labels_results[unique_positions]
            rel_labels_results = rel_labels_results[unique_positions]
            sce_labels_results = sce_labels_results[unique_positions]
            oth_labels_results = oth_labels_results[unique_positions]
            all_labels_results = all_labels_results[unique_positions]
            label_masks_results = label_masks_results[unique_positions]
            ids_results = np.array([ids_results[i] for i in unique_positions])
            image_paths_results = np.array([image_paths_results[i] for i in unique_positions])

            print("embeddings shape:", embeddings_results.shape)
            print("embeddings[0]:", embeddings_results[0])
            print("attention_masks shape:", attention_masks_results.shape)
            print("attention_masks[0]:", attention_masks_results[0])
            print("obj_labels shape:", obj_labels_results.shape)
            print("obj_labels[0]:", obj_labels_results[0])
            print("att_labels shape:", att_labels_results.shape)
            print("att_labels[0]:", att_labels_results[0])
            print("rel_labels shape:", rel_labels_results.shape)
            print("rel_labels[0]:", rel_labels_results[0])
            print("sce_labels shape:", sce_labels_results.shape)
            print("sce_labels[0]:", sce_labels_results[0])
            print("oth_labels shape:", oth_labels_results.shape)
            print("oth_labels[0]:", oth_labels_results[0])
            print("all_labels shape:", all_labels_results.shape)
            print("all_labels[0]:", all_labels_results[0])
            print("label_masks shape:", label_masks_results.shape)
            print("label_masks[0]:", label_masks_results[0])
            print("unique shape:", unique.shape)
            print("unique[0]:", unique[0])
            print("ids shape:", ids_results.shape)
            print("ids[0]:", ids_results[0])
            print("image_paths shape:", image_paths_results.shape)
            print("image_paths[0]:", image_paths_results[0])
            np.save(os.path.join(save_dir, "embeddings.npy"), embeddings_results.numpy())
            np.save(os.path.join(save_dir, "attention_masks.npy"), attention_masks_results.numpy())
            np.save(os.path.join(save_dir, "obj_labels.npy"), obj_labels_results.numpy())
            np.save(os.path.join(save_dir, "att_labels.npy"), att_labels_results.numpy())
            np.save(os.path.join(save_dir, "rel_labels.npy"), rel_labels_results.numpy())
            np.save(os.path.join(save_dir, "sce_labels.npy"), sce_labels_results.numpy())
            np.save(os.path.join(save_dir, "oth_labels.npy"), oth_labels_results.numpy())
            np.save(os.path.join(save_dir, "all_labels.npy"), all_labels_results.numpy())
            np.save(os.path.join(save_dir, "label_masks.npy"), label_masks_results.numpy())
            np.save(os.path.join(save_dir, "indices.npy"), unique)
            np.save(os.path.join(save_dir, "ids.npy"), ids_results)
            np.save(os.path.join(save_dir, "image_paths.npy"), image_paths_results)
    finally:
        accelerator.wait_for_everyone()
        accelerator.end_training()


if __name__ == "__main__":
    fire.Fire(main)
