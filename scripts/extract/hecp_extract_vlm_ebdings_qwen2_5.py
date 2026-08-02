import sys
import types
import importlib.util

# 构造空 torchaudio 占位模块，规避 CUDA 版本不兼容报错
# 本任务为纯视觉语言场景，不依赖任何音频功能
if "torchaudio" not in sys.modules:
    dummy_torchaudio = types.ModuleType("torchaudio")
    dummy_torchaudio.__version__ = "0.0.0"
    # 关键修复：必须设置 __spec__，否则 importlib.util.find_spec 会报 ValueError
    dummy_torchaudio.__spec__ = importlib.util.spec_from_loader("torchaudio", loader=None)
    sys.modules["torchaudio"] = dummy_torchaudio

import os
import gc
import json
from typing import Optional
import fire
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForImageTextToText, AutoProcessor


os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'


# ========== 新增：精准定位 assistant 回复起始位置 ==========
def find_assistant_start(input_ids, tokenizer):
    """
    在输入序列中查找 ChatML 格式的 assistant 前缀
    返回回复正文第一个 token 的索引，找不到返回 -1
    """
    assistant_prefix_ids = tokenizer.encode(
        "<|im_start|>assistant\n",
        add_special_tokens=False,
        return_tensors="pt"
    )[0].tolist()
    prefix_len = len(assistant_prefix_ids)
    seq_len = len(input_ids)
    for i in range(seq_len - prefix_len + 1):
        if input_ids[i:i+prefix_len] == assistant_prefix_ids:
            return i + prefix_len
    return -1


def read_image(image_id: str, image_dir: str, is_vg: bool) -> Image.Image:
    if is_vg:  # VISUAL GENOME
        primary_path = os.path.join(image_dir, "VG_100K", f"{image_id}.jpg")
        secondary_path = os.path.join(image_dir, "VG_100K_2", f"{image_id}.jpg")
        if os.path.exists(primary_path):
            return Image.open(primary_path).convert("RGB"), primary_path
        elif os.path.exists(secondary_path):
            return Image.open(secondary_path).convert("RGB"), secondary_path
        else:
            raise FileNotFoundError(f"Image with ID {image_id} not found in {image_dir}")
    else:  # COCO
        primary_path = os.path.join(image_dir, "train2014", f"{image_id}")
        secondary_path = os.path.join(image_dir, "val2014", f"{image_id}")
        if os.path.exists(primary_path):
            return Image.open(primary_path).convert("RGB"), primary_path
        elif os.path.exists(secondary_path):
            return Image.open(secondary_path).convert("RGB"), secondary_path
        else:
            raise FileNotFoundError(f"Image with ID not found in {primary_path}, {secondary_path}")


def load_dataset(data_path: str, image_dir: str, is_vg: bool = True):
    with open(data_path, "r") as f:
        data = json.load(f)
    dataset = []
    for idx, item in tqdm(enumerate(data)):
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
                    elif h_type == "all":
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


# --- 分片保存函数 ---
def save_chunk_results(
    chunk_id, save_dir,
    indices_results, ids_results, image_paths_results,
    hidden_states_results,
    attention_masks_results, labels_results,
    input_ids_results, token_type_ids_results,
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

    if hidden_states_merged is not None:
        np.save(os.path.join(save_dir, f"hidden_states_selected_layers{suffix}.npy"), hidden_states_merged.numpy().astype(np.float16))

    for k in labels_merged:
        if labels_merged[k] is not None:
            fname = f"{k}_labels{suffix}.npy" if k != "mask" else f"label_masks{suffix}.npy"
            np.save(os.path.join(save_dir, fname), labels_merged[k].numpy())

    if input_ids_merged is not None:
        np.save(os.path.join(save_dir, f"input_ids{suffix}.npy"), input_ids_merged.cpu().numpy())
    if token_type_ids_merged is not None:
        np.save(os.path.join(save_dir, f"token_type_ids{suffix}.npy"), token_type_ids_merged.cpu().numpy())

    np.save(os.path.join(save_dir, f"indices{suffix}.npy"), unique)
    np.save(os.path.join(save_dir, f"ids{suffix}.npy"), ids_merged)
    np.save(os.path.join(save_dir, f"image_paths{suffix}.npy"), image_paths_merged)
    print(f"[Chunk {chunk_id}] Saved successfully!")


# --- Qwen2.5-VL 前向传播，提取内部特征 + 回复起始位置 ---
def forward_qwen2_5vl_with_internals(model, processor, images, text_input, text_output, accelerator):
    device = accelerator.device
    batch_size = len(images)
    tokenizer = processor.tokenizer

    # 1. 构建 Qwen2.5-Instruct ChatML 格式的多模态对话
    conversations = []
    for prompt, text in zip(text_input, text_output):
        conv = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": [{"type": "text", "text": text}]}
        ]
        conversations.append(conv)

    # 2. 使用 apply_chat_template 将对话转换为模型所需的文本格式
    # 注意：必须设置 tokenize=False, add_generation_prompt=False
    text_prompts = processor.apply_chat_template(
        conversations, 
        tokenize=False, 
        add_generation_prompt=False
    )

    # 3. 统一编码图像与文本（明确使用关键字参数 text 和 images）
    inputs = processor(
        text=text_prompts,
        images=images,
        padding=True,
        truncation=True,
        max_length=2048,
        return_tensors="pt",
    ).to(device)

    # 模型前向推理
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            output_attentions=False,
            return_dict=True,
        )

    # 选取 1/4、1/2、3/4、最后一层（跳过 embedding 层）
    decoder_hidden_states = outputs.hidden_states[1:]
    total_layers = len(decoder_hidden_states)
    layers_to_extract = [
        (total_layers // 4) - 1,
        (total_layers // 2) - 1,
        (3 * total_layers // 4) - 1,
        total_layers - 1,
    ]

    # 计算每个样本 assistant 回复正文的起始索引
    assistant_starts = []
    input_ids_list = inputs.input_ids.cpu().tolist()
    for ids in input_ids_list:
        start = find_assistant_start(ids, tokenizer)
        if start == -1:
            raise ValueError("未找到 assistant 起始标记，对话格式异常，请检查输入")
        assistant_starts.append(start)

    # 按样本拆分隐状态
    all_hidden_states = []
    for b in range(batch_size):
        selected_hidden_states = [decoder_hidden_states[idx][b] for idx in layers_to_extract]
        layer_hidden_states = torch.stack(selected_hidden_states, dim=0)
        all_hidden_states.append(layer_hidden_states)

    # 按样本拆分 attention mask、input_ids
    all_attention_masks = [inputs.attention_mask[b] for b in range(batch_size)]
    all_input_ids = [inputs.input_ids[b] for b in range(batch_size)]

    return all_hidden_states, all_attention_masks, all_input_ids, assistant_starts


def main(
    save_dir: str,
    data_path: str,
    image_dir: str,
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    target_checkpoint: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 2,
    is_vg: bool = True,
    sample_ratio: float = 1,
):
    os.makedirs(save_dir, exist_ok=True)
    accelerator = Accelerator()
    num_gpus = torch.cuda.device_count()

    accelerator.print(f"Number of GPUs available: {num_gpus}")
    accelerator.print(f"Accelerator distributed: {accelerator.use_distributed}")
    accelerator.print("[args] model_name:", model_name)
    accelerator.print("[args] target_checkpoint:", target_checkpoint)
    accelerator.print("[args] save_dir:", save_dir)
    accelerator.print("[args] data_path:", data_path)
    accelerator.print("[args] image_dir:", image_dir)
    accelerator.print("[args] batch_size:", batch_size)
    accelerator.print("[args] num_workers:", num_workers)
    accelerator.print("[args] sample_ratio:", sample_ratio)

    # ========== 加载 Qwen2.5-VL 模型与处理器 ==========
    accelerator.print("Load model and processor")
    processor = AutoProcessor.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        attn_implementation="sdpa"
    )
    model.eval()

    accelerator.print("Load target checkpoint")
    if target_checkpoint is not None:
        model.load_state_dict(torch.load(target_checkpoint, map_location=accelerator.device), strict=False)

    accelerator.print("Define collate fn")
    def collate_fn(batch):
        index = [item["index"] for item in batch]
        id = [item["id"] for item in batch]
        images = [item["image"] for item in batch]
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
            "index": index, "id": id, "image": images, "image_path": image_path,
            "prompt": prompt, "text": text,
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

    accelerator.print("Prepare model and dataloader")
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    accelerator.print(f"Tokenizer vocab size: {processor.tokenizer.vocab_size}")
    accelerator.print("Start inference")
    
    # ========== 防止爆内存，增加计算保存间隔 (每 10% 保存一次) ==========
    total_batches = len(test_dataloader)
    save_interval = max(1, int(total_batches * 0.1))
    accelerator.print(f"Total batches: {total_batches}, Save interval: {save_interval} batches")

    indices_results, ids_results, image_paths_results = [], [], []
    hidden_states_results, attention_masks_results = [], []
    labels_results = {k: [] for k in ["obj", "att", "rel", "sce", "oth", "all", "mask"]}
    input_ids_results = []
    token_type_ids_results = []

    chunk_id = 0

    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(test_dataloader)):
                index = batch["index"]
                id = batch["id"]
                images = batch["image"]
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

                # ========== Qwen2.5-VL 前向传播 ==========
                all_hidden_states, all_attention_masks, all_input_ids, assistant_starts = \
                    forward_qwen2_5vl_with_internals(model, processor, images, text_input, text_output, accelerator)

                # ========== 精准截断：从 assistant 起始位置截取回复正文 ==========
                batch_size_cur = len(images)
                response_len_list = [torch.sum(mask == 1).item() for mask in batch["label_mask"]]

                # 截断 attention_mask / input_ids / token_type_ids
                new_attention_masks = []
                new_input_ids = []
                new_token_type_ids = []
                for b in range(batch_size_cur):
                    start = assistant_starts[b]
                    res_len = response_len_list[b]
                    end = start + res_len
                    new_attention_masks.append(all_attention_masks[b][start:end])
                    new_input_ids.append(all_input_ids[b][start:end])
                    new_token_type_ids.append(torch.zeros_like(all_input_ids[b][start:end]))

                attention_masks = pad_sequence(new_attention_masks, batch_first=True, padding_value=0)
                input_ids_padded = pad_sequence(new_input_ids, batch_first=True, padding_value=0)
                token_type_ids_padded = pad_sequence(new_token_type_ids, batch_first=True, padding_value=0)

                # 截断 hidden_states: (num_layers, seq_len, hidden_dim)
                new_all_hidden_states = []
                for b in range(batch_size_cur):
                    start = assistant_starts[b]
                    res_len = response_len_list[b]
                    end = start + res_len
                    hs = all_hidden_states[b][:, start:end, :]
                    new_all_hidden_states.append(hs.permute(1, 0, 2))
                all_hidden_states = pad_sequence(new_all_hidden_states, batch_first=True, padding_value=0)
                all_hidden_states = all_hidden_states.permute(0, 2, 1, 3)

                # ========== 多进程聚合 ==========
                attention_masks = accelerator.gather(accelerator.pad_across_processes(attention_masks, dim=1)).cpu()
                all_hidden_states = accelerator.gather(accelerator.pad_across_processes(all_hidden_states, dim=2)).cpu()

                labels_gathered = {}
                for k, v in zip(["obj", "att", "rel", "sce", "oth", "all", "mask"],
                                [obj_label, att_label, rel_label, sce_label, oth_label, all_label, label_mask]):
                    labels_gathered[k] = accelerator.gather(accelerator.pad_across_processes(v, dim=1)).cpu()

                input_ids_padded = accelerator.gather(
                    accelerator.pad_across_processes(input_ids_padded, dim=1, pad_index=0)
                ).cpu()
                token_type_ids_padded = accelerator.gather(
                    accelerator.pad_across_processes(token_type_ids_padded, dim=1, pad_index=0)
                ).cpu()

                index = accelerator.gather(torch.tensor(batch["index"], device=accelerator.device)).cpu().numpy().tolist()
                id = accelerator.gather_for_metrics(batch["id"])
                image_path = accelerator.gather_for_metrics(batch["image_path"])

                if accelerator.is_main_process:
                    attention_masks_results.append(attention_masks)
                    hidden_states_results.append(all_hidden_states)
                    for k in labels_results:
                        labels_results[k].append(labels_gathered[k])

                    input_ids_results.append(input_ids_padded)
                    token_type_ids_results.append(token_type_ids_padded)

                    indices_results.extend(index)
                    ids_results.extend(id)
                    image_paths_results.extend(image_path)

                    if (batch_idx + 1) % save_interval == 0:
                        save_chunk_results(
                            chunk_id, save_dir,
                            indices_results, ids_results, image_paths_results,
                            hidden_states_results,
                            attention_masks_results, labels_results,
                            input_ids_results, token_type_ids_results,
                        )
                        chunk_id += 1
                        indices_results.clear(); ids_results.clear(); image_paths_results.clear()
                        hidden_states_results.clear()
                        attention_masks_results.clear()
                        for k in labels_results: labels_results[k].clear()
                        input_ids_results.clear()
                        token_type_ids_results.clear()

                gc.collect()
                torch.cuda.empty_cache()

        accelerator.wait_for_everyone()
        if accelerator.is_main_process and len(indices_results) > 0:
            save_chunk_results(
                chunk_id, save_dir,
                indices_results, ids_results, image_paths_results,
                hidden_states_results,
                attention_masks_results, labels_results,
                input_ids_results, token_type_ids_results,
            )
            print("\nAll chunks saved successfully!")

    finally:
        accelerator.wait_for_everyone()
        accelerator.end_training()


if __name__ == "__main__":
    fire.Fire(main)