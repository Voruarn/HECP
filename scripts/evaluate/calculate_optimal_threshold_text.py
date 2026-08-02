import time
import json
from typing import Optional

import fire
import torch
from tqdm import tqdm
from loguru import logger
from PIL import Image

from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from src.model import get_model
from src.datamodule.dataset import get_dataset

def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Elapsed time: {time.time() - start:.2f} seconds")
        return result
    return wrapper


@timed
def calculate_optimal_threshold(
    data_dir: str,
    save_filename: str,
    checkpoint_path: Optional[str] = None,
    device: str = "cuda:0",
):
    logger.info("Load dataset")
    dataset = get_dataset("halloc_text").from_config({"file_paths": data_dir, "all_flag": False})
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=32,
        pin_memory=False,
        shuffle=False,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    logger.info("Load model")
    model = get_model("halloc_text")(
        model_name_or_path="bert-base-uncased",
        vlm_out_dim=4096,
    )
    if checkpoint_path is not None:
       # checkpoint = torch.load(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        target_prefix = "model."
        processed_checkpoint = {
            k.replace(target_prefix, ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith(target_prefix) and "position_ids" not in k
        }
        model.load_state_dict(processed_checkpoint)
    model.eval()
    model = model.to(device)

    logger.info("Start model inference")
    all_obj_labels = []
    all_att_labels = []
    all_rel_labels = []
    all_sce_labels = []
    all_oth_labels = []
    all_total_labels = []

    all_obj_probs = []
    all_att_probs = []
    all_rel_probs = []
    all_sce_probs = []
    all_oth_probs = []
    all_total_probs = []

    for batch in tqdm(dataloader):
        input_ids = batch["input_ids"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        attention_masks = batch["attention_masks"].to(device)
        image_paths = batch["image_paths"]
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]
        obj_labels = batch["obj_labels"].to(device)
        att_labels = batch["att_labels"].to(device)
        rel_labels = batch["rel_labels"].to(device)
        sce_labels = batch["sce_labels"].to(device)
        oth_labels = batch["oth_labels"].to(device)

        with torch.no_grad():
            obj_logits, att_logits, rel_logits, sce_logits, oth_logits = model(
                input_ids=input_ids,
                attention_masks=attention_masks,
                token_type_ids=token_type_ids,
                images=images,
                is_all=True,
            )

        attention_masks = attention_masks[:, 1:-1]  # (B, L)

        attention_masks = attention_masks.reshape(-1)
        nonzero_indices = attention_masks.nonzero(as_tuple=True)

        # obj_logits = obj_logits[:, :, 1]
        # obj_logits = obj_logits.view(-1)
        # obj_labels = obj_labels.view(-1)
        # filtered_obj_logits = obj_logits[nonzero_indices]
        # filtered_obj_probs = torch.softmax(filtered_obj_logits, dim=-1)
        # filtered_obj_labels = obj_labels[nonzero_indices]

        # att_logits = att_logits[:, :, 1]
        # att_logits = att_logits.view(-1)
        # att_labels = att_labels.view(-1)
        # filtered_att_logits = att_logits[nonzero_indices]
        # filtered_att_probs = torch.softmax(filtered_att_logits, dim=-1)
        # filtered_att_labels = att_labels[nonzero_indices]

        # rel_logits = rel_logits[:, :, 1]
        # rel_logits = rel_logits.view(-1)
        # rel_labels = rel_labels.view(-1)
        # filtered_rel_logits = rel_logits[nonzero_indices]
        # filtered_rel_probs = torch.softmax(filtered_rel_logits, dim=-1)
        # filtered_rel_labels = rel_labels[nonzero_indices]

        # sce_logits = sce_logits[:, :, 1]
        # sce_logits = sce_logits.view(-1)
        # sce_labels = sce_labels.view(-1)
        # filtered_sce_logits = sce_logits[nonzero_indices]
        # filtered_sce_probs = torch.softmax(filtered_sce_logits, dim=-1)
        # filtered_sce_labels = sce_labels[nonzero_indices]

        # oth_logits = oth_logits[:, :, 1]
        # oth_logits = oth_logits.view(-1)
        # oth_labels = oth_labels.view(-1)
        # filtered_oth_logits = oth_logits[nonzero_indices]
        # filtered_oth_probs = torch.softmax(filtered_oth_logits, dim=-1)
        # filtered_oth_labels = oth_labels[nonzero_indices]

        obj_logits = obj_logits.view(-1, obj_logits.size(-1))  # (B * L, C)
        obj_labels = obj_labels.view(-1)  # (B * L)
        filtered_obj_logits = obj_logits[nonzero_indices]
        filtered_obj_probs = torch.softmax(filtered_obj_logits, dim=-1)
        filtered_obj_labels = obj_labels[nonzero_indices]

        att_logits = att_logits.view(-1, att_logits.size(-1))  # (B * L, C)
        att_labels = att_labels.view(-1)  # (B * L)
        filtered_att_logits = att_logits[nonzero_indices]
        filtered_att_probs = torch.softmax(filtered_att_logits, dim=-1)
        filtered_att_labels = att_labels[nonzero_indices]

        rel_logits = rel_logits.view(-1, rel_logits.size(-1))  # (B * L, C)
        rel_labels = rel_labels.view(-1)  # (B * L)
        filtered_rel_logits = rel_logits[nonzero_indices]
        filtered_rel_probs = torch.softmax(filtered_rel_logits, dim=-1)
        filtered_rel_labels = rel_labels[nonzero_indices]

        sce_logits = sce_logits.view(-1, sce_logits.size(-1))  # (B * L, C)
        sce_labels = sce_labels.view(-1)  # (B * L)
        filtered_sce_logits = sce_logits[nonzero_indices]
        filtered_sce_probs = torch.softmax(filtered_sce_logits, dim=-1)
        filtered_sce_labels = sce_labels[nonzero_indices]

        oth_logits = oth_logits.view(-1, oth_logits.size(-1))  # (B * L, C)
        oth_labels = oth_labels.view(-1)  # (B * L)
        filtered_oth_logits = oth_logits[nonzero_indices]
        filtered_oth_probs = torch.softmax(filtered_oth_logits, dim=-1)
        filtered_oth_labels = oth_labels[nonzero_indices]

        total_probs = torch.max(filtered_obj_probs[:, 1], filtered_att_probs[:, 1])
        total_probs = torch.max(total_probs, filtered_rel_probs[:, 1])
        total_probs = torch.max(total_probs, filtered_sce_probs[:, 1])
        total_probs = torch.max(total_probs, filtered_oth_probs[:, 1])
        total_probs = torch.stack([1 - total_probs, total_probs], dim=-1)
        total_labels = filtered_obj_labels | filtered_att_labels | filtered_rel_labels | filtered_sce_labels | filtered_oth_labels

        print(filtered_obj_probs)
        print(filtered_obj_probs.shape)
        print(filtered_obj_probs.device)

        all_obj_probs.append(filtered_obj_probs.cpu())
        all_att_probs.append(filtered_att_probs.cpu())
        all_rel_probs.append(filtered_rel_probs.cpu())
        all_sce_probs.append(filtered_sce_probs.cpu())
        all_oth_probs.append(filtered_oth_probs.cpu())
        all_total_probs.append(total_probs.cpu())

        all_obj_labels.append(filtered_obj_labels.cpu())
        all_att_labels.append(filtered_att_labels.cpu())
        all_rel_labels.append(filtered_rel_labels.cpu())
        all_sce_labels.append(filtered_sce_labels.cpu())
        all_oth_labels.append(filtered_oth_labels.cpu())
        all_total_labels.append(total_labels.cpu())

    all_obj_probs = torch.cat(all_obj_probs).numpy()
    all_att_probs = torch.cat(all_att_probs).numpy()
    all_rel_probs = torch.cat(all_rel_probs).numpy()
    all_sce_probs = torch.cat(all_sce_probs).numpy()
    all_oth_probs = torch.cat(all_oth_probs).numpy()
    all_total_probs = torch.cat(all_total_probs).numpy()

    all_obj_labels = torch.cat(all_obj_labels).numpy()
    all_att_labels = torch.cat(all_att_labels).numpy()
    all_rel_labels = torch.cat(all_rel_labels).numpy()
    all_sce_labels = torch.cat(all_sce_labels).numpy()
    all_oth_labels = torch.cat(all_oth_labels).numpy()
    all_total_labels = torch.cat(all_total_labels).numpy()

    logger.info("Calculate optimal thresholds")
    thresholds = [i * 0.001 for i in range(1000)]

    best_thresholds = {}
    best_f1_scores = {}
    precision_at_best_thresholds = {}
    recall_at_best_thresholds = {}

    for task, probs, labels in tqdm(zip(
        ['obj', 'att', 'rel', 'glo', 'oth', 'total'],
        [all_obj_probs, all_att_probs, all_rel_probs, all_sce_probs, all_oth_probs, all_total_probs],
        [all_obj_labels, all_att_labels, all_rel_labels, all_sce_labels, all_oth_labels, all_total_labels]
    )):
        best_f1 = 0
        best_threshold = 0
        precision_at_best_threshold = 0
        recall_at_best_threshold = 0
        for threshold in thresholds:
            preds = (probs > threshold).astype(int)
            preds = preds[:, 1]
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
                precision_at_best_threshold = precision
                recall_at_best_threshold = recall
        best_thresholds[task] = best_threshold
        best_f1_scores[task] = best_f1
        precision_at_best_thresholds[task] = precision_at_best_threshold
        recall_at_best_thresholds[task] = recall_at_best_threshold

    result = {
        "checkpoint_path": checkpoint_path,
        "threshold": best_thresholds,
        "f1_score": best_f1_scores,
        "precision": precision_at_best_thresholds,
        "recall": recall_at_best_thresholds,
    }

    logger.info("Save optimal threshold and metrics")
    logger.info(result)
    with open(save_filename, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    fire.Fire(calculate_optimal_threshold)
