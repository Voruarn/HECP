import os
import sys
import json
from typing import Optional
from time import time

import fire
import gc
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from accelerate import Accelerator
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Add the necessary path for module imports

from internvl.conversation import get_conv_template

# Environment settings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

# Constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMG_START_TOKEN = '<img>'
IMG_END_TOKEN = '</img>'
IMG_CONTEXT_TOKEN = '<IMG_CONTEXT>'

def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=T.InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff or (ratio_diff == best_ratio_diff and area > 0.5 * image_size ** 2 * ratio[0] * ratio[1]):
            best_ratio_diff = ratio_diff
            best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # Generate target aspect ratios
    target_ratios = {(i, j) for n in range(min_num, max_num + 1)
                     for i in range(1, n + 1) for j in range(1, n + 1)
                     if min_num <= i * j <= max_num}
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # Find the closest aspect ratio
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # Calculate target dimensions
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # Resize and split the image
    resized_img = image.resize((target_width, target_height))
    processed_images = [
        resized_img.crop((
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )) for i in range(blocks)
    ]
    if use_thumbnail and blocks != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = torch.stack([transform(img) for img in images])
    return pixel_values

def get_image_path(image_id: str, image_dir: str, is_vg: bool) -> str:
    if is_vg:  # VISUAL GENOME
        paths = [
            os.path.join(image_dir, "VG_100K", f"{image_id}.jpg"),
            os.path.join(image_dir, "VG_100K_2", f"{image_id}.jpg")
        ]
    else:  # COCO
        paths = [
            os.path.join(image_dir, "train2014", image_id),
            os.path.join(image_dir, "val2014", image_id)
        ]
    for path in paths:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Image with ID {image_id} not found in {image_dir}")

def load_dataset(data_path: str, image_dir: str, is_vg: bool = True):
    with open(data_path, "r") as f:
        data = json.load(f)

    dataset = []
    for idx, item in enumerate(tqdm(data)):
        data_id = item["id"]
        image_id = item["image_id"]
        prompt = item["prompt"].replace("<image>", "").strip()
        text = item["hallucinated_text"]
        annotations = item["annotations"]
        tokenized_text = item["tokenized_text"]

        image_path = get_image_path(image_id, image_dir, is_vg)

        # Initialize token indices
        h_token_indices = {key: [] for key in ["object", "attribute", "relationship", "scene", "other", "all"]}

        # Parse annotations
        for h_type, h_list in annotations.items():
            for h in h_list:
                for k, hallucinated_answer in h.items():
                    token_index_splits = hallucinated_answer["token_index"].split(":")
                    if len(token_index_splits) == 1:
                        new_token_indices = [int(token_index_splits[0])]
                    else:
                        start, end = map(int, token_index_splits)
                        new_token_indices = list(range(start, end))
                    if h_type in h_token_indices:
                        h_token_indices[h_type].extend(new_token_indices)
                        h_token_indices["all"].extend(new_token_indices)
                    else:
                        raise ValueError(f"Unknown hallucination type: {h_type}")

        token_length = len(tokenized_text)
        labels = {key: torch.zeros(token_length, dtype=torch.long) for key in h_token_indices.keys()}
        label_mask = torch.ones(token_length, dtype=torch.long)

        # Assign labels
        for i in range(token_length):
            for h_type in h_token_indices:
                if i in h_token_indices[h_type]:
                    labels[h_type][i] = 1

        dataset.append({
            "index": idx,
            "id": data_id,
            "image_path": image_path,
            "prompt": prompt,
            "text": text,
            "labels": labels,
            "label_mask": label_mask,
        })

    return dataset

def main(
    save_dir: str,
    data_path: str,
    image_dir: str,
    target_checkpoint: Optional[str] = None,
    batch_size: int = 2,
    num_workers: int = 8,
    is_vg: bool = True,
):
    os.makedirs(save_dir, exist_ok=True)
    accelerator = Accelerator()
    num_gpus = torch.cuda.device_count()

    accelerator.print(f"Number of GPUs available: {num_gpus}")
    accelerator.print(f"Accelerator distributed: {accelerator.use_distributed}")
    accelerator.print(f"[args] target_checkpoint: {target_checkpoint}")
    accelerator.print(f"[args] save_dir: {save_dir}")
    accelerator.print(f"[args] data_path: {data_path}")
    accelerator.print(f"[args] image_dir: {image_dir}")
    accelerator.print(f"[args] batch_size: {batch_size}")
    accelerator.print(f"[args] num_workers: {num_workers}")

    accelerator.print("Load model")
    model_path = "llm/InternVL2-8B"
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
    )

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id
    num_image_token = int((448 // 14) ** 2 * (0.5 ** 2))
    accelerator.print(f"num_image_token: {num_image_token}")

    def collate_fn(batch):
        index = [item["index"] for item in batch]
        ids = [item["id"] for item in batch]
        image_paths = [item["image_path"] for item in batch]
        prompts = [item["prompt"] for item in batch]
        texts = [item["text"] for item in batch]
        labels_list = [item["labels"] for item in batch]
        label_masks = [item["label_mask"] for item in batch]

        # Load and process images
        pixel_values = [load_image(path, max_num=1).to(torch.bfloat16) for path in image_paths]
        num_patches_list = [pv.shape[0] for pv in pixel_values]
        pixel_values = torch.cat(pixel_values, dim=0)
        image_flags = torch.ones(pixel_values.size(0), dtype=torch.long)

        # Prepare queries
        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = prompts[idx].replace('<image>', '').strip()
            answer = texts[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template("internlm2-chat")
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], answer)
            query = template.get_prompt()
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        # Tokenize queries
        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        # Pad labels
        label_keys = labels_list[0].keys()
        padded_labels = {key: pad_sequence([item[key] for item in labels_list], batch_first=True, padding_value=0)
                         for key in label_keys}
        label_mask = pad_sequence(label_masks, batch_first=True, padding_value=0)

        return {
            "index": index,
            "ids": ids,
            "image_paths": image_paths,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": padded_labels,
            "label_mask": label_mask,
        }

    accelerator.print("Load dataset")
    test_dataset = load_dataset(data_path, image_dir, is_vg)
    accelerator.print(f"Total test dataset size: {len(test_dataset)}")
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=collate_fn,
    )

    accelerator.print("Prepare model and dataloader")
    model, test_dataloader = accelerator.prepare(model, test_dataloader)
    accelerator.print("Start inference")

    batch_counter = 0

    try:
        with torch.no_grad():
            for batch in tqdm(test_dataloader):
                start_time = time()
                index = batch["index"]
                ids = batch["ids"]
                image_paths = batch["image_paths"]
                pixel_values = batch["pixel_values"].to(accelerator.device)
                image_flags = batch["image_flags"].to(accelerator.device)
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)
                labels = {k: v.to(accelerator.device) for k, v in batch["labels"].items()}
                label_mask = batch["label_mask"].to(accelerator.device)

                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_flags=image_flags,
                    output_hidden_states=True,
                    return_dict=True,
                )

                # Process logits and masks
                logits = outputs.logits[:, -1 - label_mask.shape[1]:-1, :]
                input_ids = input_ids[:, -1 - label_mask.shape[1]:-1]
                attention_masks = attention_mask[:, -1 - label_mask.shape[1]:-1]

                num_zeros_in_attention_masks = [torch.sum(mask == 0).item() for mask in attention_masks]
                num_ones_in_label_masks = [torch.sum(mask == 1).item() for mask in label_mask]

                # Trim tensors based on masks
                new_logits, new_input_ids, new_attention_masks = [], [], []
                for logit, input_id, attn_mask, zeros_in_attn, ones_in_label in zip(
                        logits, input_ids, attention_masks, num_zeros_in_attention_masks, num_ones_in_label_masks):
                    start = -(zeros_in_attn + ones_in_label)
                    end = None if zeros_in_attn == 0 else -zeros_in_attn
                    new_logits.append(logit[start:end])
                    new_input_ids.append(input_id[start:end])
                    new_attention_masks.append(attn_mask[start:end])

                # Pad sequences
                logits = pad_sequence(new_logits, batch_first=True, padding_value=0)
                input_ids = pad_sequence(new_input_ids, batch_first=True, padding_value=0)
                attention_masks = pad_sequence(new_attention_masks, batch_first=True, padding_value=0)

                # Compute probabilities
                log_probs = F.softmax(logits, dim=-1)
                batch_size, seq_len, vocab_size = log_probs.shape
                expanded_attention_masks = attention_masks.unsqueeze(-1)
                masked_log_probs = log_probs * expanded_attention_masks
                log_probs_of_input_ids = masked_log_probs[
                    torch.arange(batch_size).unsqueeze(1), torch.arange(seq_len), input_ids]

                # Apply attention masks
                masked_logits = logits * expanded_attention_masks
                logits = masked_logits.view(batch_size, seq_len, vocab_size)

                # Move tensors to CPU
                logits = logits.cpu()
                log_probs_of_input_ids = log_probs_of_input_ids.cpu()
                input_ids = input_ids.cpu()
                attention_masks = attention_masks.cpu()
                labels = {k: v.cpu() for k, v in labels.items()}
                label_mask = label_mask.cpu()

                # Increment batch counter
                batch_counter += 1

                # Generate unique filename prefix
                filename_prefix = f"rank{accelerator.process_index}_batch{batch_counter}"

                # Save the batch data
                np.save(os.path.join(save_dir, f"logits_{filename_prefix}.npy"), logits.numpy())
                np.save(os.path.join(save_dir, f"probs_{filename_prefix}.npy"), log_probs_of_input_ids.numpy())
                np.save(os.path.join(save_dir, f"input_ids_{filename_prefix}.npy"), input_ids.numpy())
                np.save(os.path.join(save_dir, f"attention_masks_{filename_prefix}.npy"), attention_masks.numpy())
                for label_name, label_tensor in labels.items():
                    np.save(os.path.join(save_dir, f"{label_name}_labels_{filename_prefix}.npy"), label_tensor.numpy())
                np.save(os.path.join(save_dir, f"label_masks_{filename_prefix}.npy"), label_mask.numpy())
                np.save(os.path.join(save_dir, f"indices_{filename_prefix}.npy"), np.array(index))
                np.save(os.path.join(save_dir, f"ids_{filename_prefix}.npy"), np.array(ids))
                np.save(os.path.join(save_dir, f"image_paths_{filename_prefix}.npy"), np.array(image_paths))

                accelerator.print(f"Batch {batch_counter} processed in {time() - start_time:.2f}s")

        accelerator.wait_for_everyone()

    finally:
        accelerator.end_training()

if __name__ == "__main__":
    fire.Fire(main)
