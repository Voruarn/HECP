import os
import sys
import gc
import json
from typing import Optional
from time import time

import fire
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import torch.nn.functional as F
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from transformers import AutoTokenizer, AutoModel

from internvl.conversation import get_conv_template

os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def get_image_path(image_id: str, image_dir: str, is_vg: bool) -> Image.Image:
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
    for idx, item in tqdm(enumerate(data)):
        data_id = item["id"]
        image_id = item["image_id"]
        prompt = item["prompt"]
        text = item["hallucinated_text"]
        annotations = item["annotations"]
        tokenized_text = item["tokenized_text"]

        prompt = prompt.replace("<image>", "").strip()

        image_path = get_image_path(image_id, image_dir, is_vg)

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

    accelerator.print("[args] target_checkpoint:", target_checkpoint)
    accelerator.print("[args] save_dir:", save_dir)
    accelerator.print("[args] data_path:", data_path)
    accelerator.print("[args] image_dir:", image_dir)
    accelerator.print("[args] batch_size:", batch_size)
    accelerator.print("[args] num_workers:", num_workers)

    accelerator.print("Load model")

    path = "llm/InternVL2-8B"
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        # low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
    ).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        path,
        trust_remote_code=True,
        use_fast=False,
    )

    IMG_START_TOKEN='<img>'
    IMG_END_TOKEN='</img>'
    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>'

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    num_image_token = int((448 // 14) ** 2 * (0.5 ** 2))

    accelerator.print("num_image_token:", num_image_token)

    accelerator.print("Define collate fn")

    def collate_fn(batch):
        s = time()
        index = [item["index"] for item in batch]
        id = [item["id"] for item in batch]
        
        pixel_values = [load_image(item["image_path"], max_num=1).to(torch.bfloat16) for item in batch]
        num_patches_list = [pixel_value.shape[0] for pixel_value in pixel_values]
        pixel_values = torch.cat(pixel_values, dim=0) # VERIFY
        assert len(pixel_values) == sum(num_patches_list)
        image_flags=torch.tensor([1] * pixel_values.size(0), dtype=torch.long)
        
        image_path = [item["image_path"] for item in batch]

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        img_context_token_id = img_context_token_id
        
        queries = []
        questions = [item["prompt"] for item in batch]
        answers = [item["text"] for item in batch]
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            # VERIFY
            question = question.replace('<image>', '').strip()
            answer = answers[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template("internlm2-chat")
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], answer)
            query = template.get_prompt()

            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)
        
        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        input_ids = model_inputs['input_ids']
        attention_mask = model_inputs['attention_mask']

        obj_label = pad_sequence([item["obj_label"] for item in batch], batch_first=True, padding_value=0)
        att_label = pad_sequence([item["att_label"] for item in batch], batch_first=True, padding_value=0)
        rel_label = pad_sequence([item["rel_label"] for item in batch], batch_first=True, padding_value=0)
        sce_label = pad_sequence([item["sce_label"] for item in batch], batch_first=True, padding_value=0)
        oth_label = pad_sequence([item["oth_label"] for item in batch], batch_first=True, padding_value=0)
        all_label = pad_sequence([item["all_label"] for item in batch], batch_first=True, padding_value=0)
        label_mask = pad_sequence([item["label_mask"] for item in batch], batch_first=True, padding_value=0)

        accelerator.print(f"collate_fn time: {time()-s:.2f}s")

        return {
            "index": index,
            "id": id,
            "image_path": image_path,
            "pixel_values": pixel_values,
            "image_flags": image_flags,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
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
                s = time()
                index = batch["index"]
                id = batch["id"]
                image_path = batch["image_path"]
                pixel_values = batch["pixel_values"].to(accelerator.device)
                image_flags = batch["image_flags"].to(accelerator.device)
                input_ids = batch["input_ids"].to(accelerator.device)
                attention_mask = batch["attention_mask"].to(accelerator.device)
                obj_label = batch["obj_label"].to(accelerator.device)
                att_label = batch["att_label"].to(accelerator.device)
                rel_label = batch["rel_label"].to(accelerator.device)
                sce_label = batch["sce_label"].to(accelerator.device)
                oth_label = batch["oth_label"].to(accelerator.device)
                all_label = batch["all_label"].to(accelerator.device)
                label_mask = batch["label_mask"].to(accelerator.device)
                
                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_flags=image_flags,
                    output_hidden_states=True,
                    return_dict=True,
                )

                embeddings = outputs.hidden_states[-1]
                embeddings = embeddings[:, -1-label_mask.shape[1]:-1, :]
                attention_masks = attention_mask[:, -1-label_mask.shape[1]:-1]

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

                # gc.collect()
                # torch.cuda.empty_cache()

                print(f"batch time: {time()-s:.2f}s")

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

            np.save(os.path.join(save_dir, "embeddings.npy"), embeddings_results.to(torch.float32).numpy())
            np.save(os.path.join(save_dir, "attention_masks.npy"), attention_masks_results.to(torch.float32).numpy())
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
