import os
import sys
current_working_dir = os.getcwd()
sys.path.append(current_working_dir + '/LLaVA')
import gc
import json
from typing import Optional
import fire
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from accelerate import Accelerator
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from llava.utils import disable_torch_init
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from PIL import Image

def forward_ddp(model_name, tokenizer, accelerator, model, image_processor, args):
    # 处理输入图片
    image_files = args.image_file.split(',')
    images = []
    for image_file in image_files:
        image = Image.open(image_file).convert('RGB')
        images.append(image)
    
    images = process_images(images, image_processor, model.config)
    
    if type(images) is list:
        images = [image.to(accelerator.device, dtype=torch.float16) for image in images]
    else:
        images = images.to(accelerator.device, dtype=torch.float16)
    
    queries = args.query if isinstance(args.query, list) else [args.query]
    responses = args.response if isinstance(args.response, list) else [args.response]
    
    all_hidden_states = []
    all_attention_masks = []
    all_input_ids = []
  
    for i, (query, response) in enumerate(zip(queries, responses)):
        prompt = query
        
        if model.config.mm_use_im_start_end:
            prompt = prompt.replace('<image-placeholder>', DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN)
        else:
            prompt = prompt.replace('<image-placeholder>', DEFAULT_IMAGE_TOKEN)
        
        full_text = prompt + response
        
        input_ids = tokenizer_image_token(
            full_text, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
        ).unsqueeze(0).to(accelerator.device)
        attention_mask = torch.ones_like(input_ids).to(accelerator.device)
        
        image_tensor = images[i].unsqueeze(0) if len(images.shape) == 3 else images[i:i+1]
       
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                images=image_tensor,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
        
        decoder_hidden_states = outputs.hidden_states[1:]
        total_layers = len(decoder_hidden_states)
        layers_to_extract = [
            (total_layers // 4) - 1,
            (total_layers // 2) - 1,
            (3 * total_layers // 4) - 1,
            total_layers - 1
        ]
        selected_hidden_states = [decoder_hidden_states[idx][0] for idx in layers_to_extract]
        layer_hidden_states = torch.stack(selected_hidden_states, dim=0)
        all_hidden_states.append(layer_hidden_states)
        all_attention_masks.append(attention_mask[0])
        all_input_ids.append(input_ids[0])
    
    return all_hidden_states, all_attention_masks, all_input_ids

from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

def to_image_path(image_id: str, image_dir: str, is_vg: bool):
    if is_vg: # VISUAL GENOME
        primary_path = os.path.join(image_dir, "VG_100K", f"{image_id}.jpg")
        secondary_path = os.path.join(image_dir, "VG_100K_2", f"{image_id}.jpg")
        if os.path.exists(primary_path):
            return primary_path
        elif os.path.exists(secondary_path):
            return secondary_path
        else:
            raise FileNotFoundError(f"Image with ID {image_id} not found in {image_dir}")
    else: # COCO
        primary_path = os.path.join(image_dir, "train2014", f"{image_id}")
        secondary_path = os.path.join(image_dir, "val2014", f"{image_id}")
        if os.path.exists(primary_path):
            return primary_path
        elif os.path.exists(secondary_path):
            return secondary_path
        else:
            raise FileNotFoundError(f"Image with ID not found in {primary_path}, {secondary_path}")

def load_dataset(data_path: str, image_dir: str, is_vg: bool = True):
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
        prompt = prompt.replace('<image>', '<image-placeholder>')
        obj_h_token_indices, att_h_token_indices, rel_h_token_indices = [], [], []
        sce_h_token_indices, oth_h_token_indices, all_h_token_indices = [], [], []
        
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
                        obj_h_token_indices.extend(new_token_indices); all_h_token_indices.extend(new_token_indices)
                    elif h_type == "attribute":
                        att_h_token_indices.extend(new_token_indices); all_h_token_indices.extend(new_token_indices)
                    elif h_type == "relationship":
                        rel_h_token_indices.extend(new_token_indices); all_h_token_indices.extend(new_token_indices)
                    elif h_type == "scene":
                        sce_h_token_indices.extend(new_token_indices); all_h_token_indices.extend(new_token_indices)
                    elif h_type == "other":
                        oth_h_token_indices.extend(new_token_indices); all_h_token_indices.extend(new_token_indices)
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
            if i in obj_h_token_indices: obj_label[i] = 1
            if i in att_h_token_indices: att_label[i] = 1
            if i in rel_h_token_indices: rel_label[i] = 1
            if i in sce_h_token_indices: sce_label[i] = 1
            if i in oth_h_token_indices: oth_label[i] = 1
            if i in all_h_token_indices: all_label[i] = 1

        dataset.append({
            "index": idx, "id": data_id,
            "image_path": to_image_path(image_id, image_dir, is_vg),
            "prompt": prompt, "text": text,
            "obj_label": obj_label, "att_label": att_label, "rel_label": rel_label,
            "sce_label": sce_label, "oth_label": oth_label, "all_label": all_label,
            "label_mask": label_mask,
        })
    return dataset

# --- 辅助函数：合并不同 seq_len 的 tensor ---
def merge_results(tensor_list, seq_dim):
    if not tensor_list:
        return None
    max_seq_len = max(t.shape[seq_dim] for t in tensor_list)
    padded_list = []
    for t in tensor_list:
        current_seq_len = t.shape[seq_dim]
        pad_size = max_seq_len - current_seq_len
        if pad_size > 0:
            pad_tuple = [0, 0] * t.ndim
            pad_idx = (t.ndim - 1 - seq_dim) * 2
            pad_tuple[pad_idx + 1] = pad_size 
            t = torch.nn.functional.pad(t, tuple(pad_tuple), value=0)
        padded_list.append(t)
    return torch.cat(padded_list, dim=0)

# --- 核心修改：分片保存函数 ---
def save_chunk_results(
    chunk_id, save_dir,
    indices_results, ids_results, image_paths_results,
    hidden_states_results, input_ids_results, token_type_ids_results,
    attention_masks_results, labels_results
):
    if not indices_results:
        return

    print(f"\n[Chunk {chunk_id}] Merging and saving {len(indices_results)} samples...")
    
    indices_array = np.array(indices_results)
    unique, unique_positions = np.unique(indices_array, return_index=True)
    
    attention_masks_merged = merge_results(attention_masks_results, seq_dim=1)[unique_positions] if attention_masks_results else None
    hidden_states_merged = merge_results(hidden_states_results, seq_dim=2)[unique_positions] if hidden_states_results else None
    input_ids_merged = merge_results(input_ids_results, seq_dim=1)[unique_positions] if input_ids_results else None
    token_type_ids_merged = merge_results(token_type_ids_results, seq_dim=1)[unique_positions] if token_type_ids_results else None
    
    labels_merged = {}
    for k in labels_results:
        labels_merged[k] = merge_results(labels_results[k], seq_dim=1)[unique_positions] if labels_results[k] else None
            
    ids_merged = np.array([ids_results[i] for i in unique_positions])
    image_paths_merged = np.array([image_paths_results[i] for i in unique_positions])

    suffix = f"_chunk_{chunk_id}"
    
    if attention_masks_merged is not None:
        np.save(os.path.join(save_dir, f"attention_masks{suffix}.npy"), attention_masks_merged.numpy())
        
    # 使用 npy 保存 hidden_states
    if hidden_states_merged is not None:
        np.save(os.path.join(save_dir, f"hidden_states_selected_layers{suffix}.npy"), hidden_states_merged.numpy().astype(np.float16))
        
    if input_ids_merged is not None:
        np.save(os.path.join(save_dir, f"input_ids{suffix}.npy"), input_ids_merged.numpy())
        
    if token_type_ids_merged is not None:
        np.save(os.path.join(save_dir, f"token_type_ids{suffix}.npy"), token_type_ids_merged.numpy())
               
    for k in labels_merged:
        if labels_merged[k] is not None:
            fname = f"{k}_labels{suffix}.npy" if k != "mask" else f"label_masks{suffix}.npy"
            np.save(os.path.join(save_dir, fname), labels_merged[k].numpy())
    
    np.save(os.path.join(save_dir, f"indices{suffix}.npy"), unique)
    np.save(os.path.join(save_dir, f"ids{suffix}.npy"), ids_merged)
    np.save(os.path.join(save_dir, f"image_paths{suffix}.npy"), image_paths_merged)
    
    print(f"[Chunk {chunk_id}] Saved successfully!")


def main(
    target_checkpoint: Optional[str] = None,
    save_dir: str = "data/mhaldetect/llava/train",
    data_path: str = "json_data/mhaldetect_train_haloc_char_index_llava_postprocessed.json",
    image_dir: str = "img_data/vg_images",
    batch_size: int = 4,
    num_workers: int = 4,
    is_vg: bool = True,
    sample_ratio: float = 1,
):
    os.makedirs(save_dir, exist_ok=True)
    accelerator = Accelerator()
    accelerator.print("[args] target_checkpoint:", target_checkpoint)
    accelerator.print("[args] save_dir:", save_dir)
    
    accelerator.print("Load model")
    model_path = "liuhaotian/llava-v1.5-7b"
    model = LlavaLlamaForCausalLM.from_pretrained(model_path, attn_implementation="eager")
    model = model.to(device=accelerator.device, dtype=torch.float16)
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    model.resize_token_embeddings(len(tokenizer))
    model.eval()
    disable_torch_init()
    
    model_name = get_model_name_from_path(model_path)
    tokenizer, _, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map="cpu", device="cpu"
    )
    
    device_map = accelerator.device
    vision_tower = model.get_vision_tower()
    if not vision_tower.is_loaded:
        vision_tower.load_model(device_map=device_map)
    vision_tower.to(device=device_map, dtype=torch.float16)
    
    args_dict = {
        "query": None, "response": None, "conv_mode": None,
        "image_file": None, "sep": ",",
    }
    
    if target_checkpoint is not None:
        accelerator.print("Load target checkpoint")
        model.load_state_dict(torch.load(target_checkpoint, map_location=accelerator.device), strict=False)

    def collate_fn(batch):
        index = [item["index"] for item in batch]
        id = [item["id"] for item in batch]
        image_path = [item["image_path"] for item in batch]
        prompt = [item["prompt"] for item in batch]
        text = [item["text"] for item in batch]
        obj_label = pad_sequence([item["obj_label"] for item in batch], batch_first=True, padding_value=0)
        att_label = pad_sequence([item["att_label"] for item in batch], batch_first=True, padding_value=0)
        rel_label = pad_sequence([item["rel_label"] for item in batch], batch_first=True, padding_value=0)
        sce_label = pad_sequence([item["sce_label"] for item in batch], batch_first=True, padding_value=0)
        oth_label = pad_sequence([item["oth_label"] for item in batch], batch_first=True, padding_value=0)
        all_label = pad_sequence([item["all_label"] for item in batch], batch_first=True, padding_value=0)
        label_mask = pad_sequence([item["label_mask"] for item in batch], batch_first=True, padding_value=0)
        return {
            "index": index, "id": id, "image_path": image_path, "prompt": prompt, "text": text,
            "obj_label": obj_label, "att_label": att_label, "rel_label": rel_label,
            "sce_label": sce_label, "oth_label": oth_label, "all_label": all_label, "label_mask": label_mask,
        }

    accelerator.print("Load dataset")
    test_dataset = load_dataset(data_path, image_dir, is_vg)
    total_samples = len(test_dataset)
    accelerator.print(f"Total test dataset size: {total_samples}")

    sample_count = max(1, int(total_samples * sample_ratio))
    test_dataset = test_dataset[:sample_count]
    accelerator.print(f"Extracted first {sample_ratio*100:.1f}% of dataset: {len(test_dataset)} samples")

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    accelerator.print("Start inference")
    
    # ========== 防止爆内存，增加计算保存间隔 (每 10% 保存一次) ==========
    total_batches = len(test_dataloader)
    save_interval = max(1, int(total_batches * 0.1))
    accelerator.print(f"Total batches: {total_batches}, Save interval: {save_interval} batches (approx 10%)")
    
    indices_results, ids_results, image_paths_results = [], [], []
    hidden_states_results, attention_masks_results = [], []
    input_ids_results, token_type_ids_results = [], []
    labels_results = {k: [] for k in ["obj", "att", "rel", "sce", "oth", "all", "mask"]}
    
    chunk_id = 0

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader)):
                args_dict['image_file'] = ",".join(batch["image_path"])
                args_dict['query'] = batch["prompt"]
                args_dict['response'] = batch["text"]
                args = type('Args', (), args_dict)()
                
                all_hidden_states, attention_masks, all_input_ids = forward_ddp(
                    model_name, tokenizer, accelerator, model, image_processor, args
                )
                
                num_zeros_in_attention_masks = [torch.sum(mask == 0).item() for mask in attention_masks]
                num_ones_in_label_masks = [torch.sum(batch["label_mask"][i] == 1).item() for i in range(len(batch["label_mask"]))]
                
                new_attention_masks = []
                for tensor, num_zeros, num_ones in zip(attention_masks, num_zeros_in_attention_masks, num_ones_in_label_masks):
                    start = -(num_zeros + num_ones)
                    end = -num_zeros if num_zeros > 0 else None
                    new_attention_masks.append(tensor[start:end])
                attention_masks = pad_sequence(new_attention_masks, batch_first=True, padding_value=0)

                new_all_hidden_states = []
                for hs, num_zeros, num_ones in zip(all_hidden_states, num_zeros_in_attention_masks, num_ones_in_label_masks):
                    start = -(num_zeros + num_ones)
                    end = -num_zeros if num_zeros > 0 else None
                    new_all_hidden_states.append(hs[:, start:end, :].permute(1, 0, 2))
                all_hidden_states = pad_sequence(new_all_hidden_states, batch_first=True, padding_value=0)
                all_hidden_states = all_hidden_states.permute(0, 2, 1, 3)

                # 增加提取 input_ids 和 token_type_ids 的截取与填充逻辑
                new_all_input_ids = []
                new_token_type_ids = []
                for ids, num_zeros, num_ones in zip(all_input_ids, num_zeros_in_attention_masks, num_ones_in_label_masks):
                    start = -(num_zeros + num_ones)
                    end = -num_zeros if num_zeros > 0 else None
                    cropped_ids = ids[start:end]
                    new_all_input_ids.append(cropped_ids)
                    new_token_type_ids.append(torch.zeros_like(cropped_ids))
                all_input_ids = pad_sequence(new_all_input_ids, batch_first=True, padding_value=0)
                token_type_ids = pad_sequence(new_token_type_ids, batch_first=True, padding_value=0)

                obj_label = batch["obj_label"].to(accelerator.device)
                att_label = batch["att_label"].to(accelerator.device)
                rel_label = batch["rel_label"].to(accelerator.device)
                sce_label = batch["sce_label"].to(accelerator.device)
                oth_label = batch["oth_label"].to(accelerator.device)
                all_label = batch["all_label"].to(accelerator.device)
                label_mask = batch["label_mask"].to(accelerator.device)

                attention_masks = accelerator.gather(accelerator.pad_across_processes(attention_masks, dim=1)).cpu()
                all_hidden_states = accelerator.gather(accelerator.pad_across_processes(all_hidden_states, dim=2)).cpu()
                all_input_ids = accelerator.gather(accelerator.pad_across_processes(all_input_ids, dim=1)).cpu()
                token_type_ids = accelerator.gather(accelerator.pad_across_processes(token_type_ids, dim=1)).cpu()

                labels_gathered = {}
                for k, v in zip(["obj", "att", "rel", "sce", "oth", "all", "mask"], 
                                [obj_label, att_label, rel_label, sce_label, oth_label, all_label, label_mask]):
                    labels_gathered[k] = accelerator.gather(accelerator.pad_across_processes(v, dim=1)).cpu()

                index = accelerator.gather(torch.tensor(batch["index"], device=accelerator.device)).cpu().numpy().tolist()
                id = accelerator.gather_for_metrics(batch["id"])
                image_path = accelerator.gather_for_metrics(batch["image_path"])

                if accelerator.is_main_process:
                    attention_masks_results.append(attention_masks)
                    hidden_states_results.append(all_hidden_states)
                    input_ids_results.append(all_input_ids)
                    token_type_ids_results.append(token_type_ids)
                        
                    for k in labels_results:
                        labels_results[k].append(labels_gathered[k])

                    indices_results.extend(index)
                    ids_results.extend(id)
                    image_paths_results.extend(image_path)

                    # ========== 核心修改：达到 10% 进度时触发保存并清空缓存 ==========
                    if (batch_idx + 1) % save_interval == 0:
                        save_chunk_results(
                            chunk_id, save_dir,
                            indices_results, ids_results, image_paths_results,
                            hidden_states_results, input_ids_results, token_type_ids_results,
                            attention_masks_results, labels_results
                        )
                        chunk_id += 1
                        # 清空当前缓存，释放内存
                        indices_results.clear(); ids_results.clear(); image_paths_results.clear()
                        hidden_states_results.clear()
                        attention_masks_results.clear()
                        input_ids_results.clear(); token_type_ids_results.clear()
                        for k in labels_results: labels_results[k].clear()

                gc.collect()
                torch.cuda.empty_cache()
                
        accelerator.wait_for_everyone()
        
        # ========== 核心修改：保存最后剩余不足 10% 的尾部数据 ==========
        if accelerator.is_main_process and len(indices_results) > 0:
            save_chunk_results(
                chunk_id, save_dir,
                indices_results, ids_results, image_paths_results,
                hidden_states_results, input_ids_results, token_type_ids_results,
                attention_masks_results, labels_results
            )
            print("\nAll chunks saved successfully!")

    finally:
        accelerator.wait_for_everyone()
        accelerator.end_training()

if __name__ == "__main__":
    fire.Fire(main)