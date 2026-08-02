import time
import json
from typing import Optional
import numpy as np
import os

import fire
import torch
from tqdm import tqdm
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, auc

from src.model import get_model
from src.datamodule.dataset import get_dataset


def pad_to_max_len(batch_data):
    """
    Pad a batch of variable-length sequences to the max length in the batch.
    
    Args:
        batch_data (List[List[List[float]]]): Input data of shape [B, ?, 2]
    
    Returns:
        torch.Tensor: Padded tensor of shape [B, max_len, 2]
    """
    # Calculate the maximum length in the batch
    max_len = max(len(seq) for seq in batch_data)
    
    # Create a padded tensor filled with zeros
    padded_tensor = torch.zeros((len(batch_data), max_len, 2), dtype=torch.float32)
    
    # Fill the padded tensor with the values from batch_data
    for i, seq in enumerate(batch_data):
        seq_len = len(seq)
        padded_tensor[i, :seq_len, :] = torch.tensor(seq, dtype=torch.float32)
    
    return padded_tensor


def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        logger.info(f"Elapsed time: {time.time() - start:.2f} seconds")
        return result
    return wrapper


@timed
def evaluate(
    threshold_path: str,
    checkpoint_path: Optional[str] = '/halloc/best.ckpt',
    device: str = "cuda:0",
    data_dir: str = '/data/val/vlm_embeddings/llava',
):
    
    with open(threshold_path, 'r') as f:
        thresholds = json.load(f)
    obj_threshold = thresholds['threshold']['obj']
    att_threshold = thresholds['threshold']['att']
    rel_threshold = thresholds['threshold']['rel']
    sce_threshold = thresholds['threshold']['glo']
    oth_threshold = thresholds['threshold']['oth']
    total_threshold = thresholds['threshold']['total']

    logger.info("Load dataset")
    dataset = get_dataset("halloc").from_config({"file_paths": data_dir, "all_flag": False})
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        num_workers=16,
        pin_memory=False,
        shuffle=False,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    logger.info("Load model")
    model = get_model("halloc")(
        model_name_or_path="bert-base-uncased",
        vlm_out_dim=4096,
    )
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path)
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

    all_obj_logits = []
    all_att_logits = []
    all_rel_logits = []
    all_sce_logits = []
    all_oth_logits = []
    all_total_logits = []
    
    all_obj_probs = []
    all_att_probs = []
    all_rel_probs = []
    all_sce_probs = []
    all_oth_probs = []
    all_total_probs = []

    all_real_lengths = []

    for i, batch in tqdm(enumerate(dataloader)):
        # if i == 2:
        #     break
        embeddings = batch["embeddings"].to(device)
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
                embeddings=embeddings,
                attention_masks=attention_masks,
                images=images,
                is_all=False,
            )

        bs = attention_masks.size(0)
        sequence_length = attention_masks.size(1)
        print("Attention Masks Shape Before: ", attention_masks.shape)
        print("Batch Size: ", bs)
        print("Max Sequence Length: ", sequence_length)

        attention_masks = attention_masks.view(-1)
        nonzero_indices = attention_masks.nonzero(as_tuple=True)

        print("Attention Masks Shape Flattened: ", attention_masks.shape)

        iterations = 0
        for i in range(0, bs * sequence_length, sequence_length):
            all_real_lengths.append(int(sum(attention_masks[i:i + sequence_length].cpu().numpy())))
            iterations += 1
        assert iterations == bs

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

        all_obj_probs.append(filtered_obj_probs.cpu())
        all_att_probs.append(filtered_att_probs.cpu())
        all_rel_probs.append(filtered_rel_probs.cpu())
        all_sce_probs.append(filtered_sce_probs.cpu())
        all_oth_probs.append(filtered_oth_probs.cpu())
        all_total_probs.append(total_probs.cpu())

        all_obj_logits.append(filtered_obj_logits.cpu())
        all_att_logits.append(filtered_att_logits.cpu())
        all_rel_logits.append(filtered_rel_logits.cpu())
        all_sce_logits.append(filtered_sce_logits.cpu())
        all_oth_logits.append(filtered_oth_logits.cpu())

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

    all_obj_logits = torch.cat(all_obj_logits).numpy()
    all_att_logits = torch.cat(all_att_logits).numpy()
    all_rel_logits = torch.cat(all_rel_logits).numpy()
    all_sce_logits = torch.cat(all_sce_logits).numpy()
    all_oth_logits = torch.cat(all_oth_logits).numpy()

    all_obj_labels = torch.cat(all_obj_labels).numpy()
    all_att_labels = torch.cat(all_att_labels).numpy()
    all_rel_labels = torch.cat(all_rel_labels).numpy()
    all_sce_labels = torch.cat(all_sce_labels).numpy()
    all_oth_labels = torch.cat(all_oth_labels).numpy()
    all_total_labels = torch.cat(all_total_labels).numpy()

    pr_aucs = {}
    best_thresholds = {}
    best_f1_scores = {}
    precision_at_best_thresholds = {}
    recall_at_best_thresholds = {}
    logits_total = {}
    preds_total = {}
    labels_total ={}

    for task, logits, probs, labels, threshold in tqdm(zip(
        ['obj', 'att', 'rel', 'glo', 'oth', 'total'],
        [all_obj_logits, all_att_logits, all_rel_logits, all_sce_logits, all_oth_logits, all_total_logits],
        [all_obj_probs, all_att_probs, all_rel_probs, all_sce_probs, all_oth_probs, all_total_probs],
        [all_obj_labels, all_att_labels, all_rel_labels, all_sce_labels, all_oth_labels, all_total_labels],
        [obj_threshold, att_threshold, rel_threshold, sce_threshold, oth_threshold, total_threshold]
    )):
        # Compute precision-recall curve
        positive_probs = probs[:, 1]
        precision, recall, thresholds = precision_recall_curve(labels, positive_probs)
        # Compute Precision-Recall AUC
        pr_auc = auc(recall, precision)
        pr_aucs[task] = pr_auc
        
        preds = (probs > threshold).astype(int)
        preds = preds[:, 1]
        print(probs.shape)
        print(preds.shape)
        print(labels.shape)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
        best_thresholds[task] = threshold
        best_f1_scores[task] = f1
        precision_at_best_thresholds[task] = precision
        recall_at_best_thresholds[task] = recall

        split_logits = []
        split_preds = []
        split_labels = []
        cnt = 0
        for real_length in all_real_lengths:
            split_logits.append(logits[cnt:cnt+real_length])
            split_preds.append(preds[cnt:cnt+real_length].tolist())
            split_labels.append(labels[cnt:cnt+real_length].tolist())
            cnt += real_length
        
        assert cnt == len(preds), f"{cnt} != {len(preds)}"
        
        if task != 'total':
            split_logits = pad_to_max_len(split_logits).numpy()
            print(split_logits.shape)
            logits_total[task] = split_logits # [B, max_len, 2)
        preds_total[task] = split_preds
        labels_total[task] = split_labels

    result = {
        "checkpoint_path": checkpoint_path,
        "pr_auc": pr_aucs,
        "threshold": best_thresholds,
        "f1_score": best_f1_scores,
        "precision": precision_at_best_thresholds,
        "recall": recall_at_best_thresholds,
        "preds": preds_total,
        "labels": labels_total,
    }

    base_dir, vlm = data_dir.split("/vlm_embeddings/")
    ckpt_version = checkpoint_path.split("/")[-3]
    
    results_dir = base_dir.replace("data", "evaluation/results")
    results_dir = os.path.join(results_dir, vlm)
    results_dir = os.path.join(results_dir, ckpt_version)

    print(f"Save dir: {results_dir}")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_filename = threshold_path.split("/")[-1].replace("thresholds", "results")
    save_filename = os.path.join(results_dir, results_filename)

    print(f"Save to {save_filename}")
    logger.info("Save optimal threshold and metrics")
    
    with open(save_filename, "w") as f:
        json.dump(result, f, indent=4)
    
    logits_dir = base_dir.replace("data", "evaluation/logits")
    logits_dir = os.path.join(logits_dir, vlm)
    logits_dir = os.path.join(logits_dir, ckpt_version)
    
    print(f"Save logits to {logits_dir}")
    if not os.path.exists(logits_dir):
        os.makedirs(logits_dir)
    
    for k, v in logits_total.items():
        print(f"Saving logits for {k} to {os.path.join(logits_dir, f'halloc_{k}_logits.npy')}")
        print(v.shape)
        np.save(os.path.join(logits_dir, f'halloc_{k}_logits.npy'), v)

if __name__ == "__main__":
    fire.Fire(evaluate)
