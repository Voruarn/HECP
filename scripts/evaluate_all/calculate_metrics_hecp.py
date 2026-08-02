import time
import json
from typing import Optional
import os

import fire
import torch
from tqdm import tqdm
from loguru import logger
from PIL import Image

from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score, average_precision_score

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
    vlm_type: str = "llava_mrg",
    task_name: str = "all",  # 修复 1: 增加 task_name 参数，匹配 __main__ 中的调用
):
    logger.info(f"Processing VLM: {vlm_type}, Task: {task_name}, Data Dir: {data_dir}")
    
    logger.info("Load dataset")
    # 将 all_flag 设置为 True 以加载合并后的标签
    dataset = get_dataset("hecp").from_config({"file_paths": data_dir, "all_flag": True})
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        num_workers=32,
        pin_memory=False,
        shuffle=False,
    )
    logger.info(f"Dataset size: {len(dataset)}")

    logger.info("Load model")
    vlm_out_dim = 4096
    if vlm_type == "qwen2_5_mrg":
        vlm_out_dim = 3584 
        
    model = get_model("hecp")(
        vlm_out_dim=vlm_out_dim,
    )
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        target_prefix = "model."
        processed_checkpoint = {
            k.replace(target_prefix, ""): v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith(target_prefix) and "position_ids" not in k
        }
        # 兼容 LoRA 权重缺少 base model 参数的情况
        model.load_state_dict(processed_checkpoint, strict=False)
    model.eval()
    model = model.to(device)

    logger.info("Start model inference")
    
    # 初始化用于存储 'all' 任务预测结果和标签的列表
    all_all_probs = []
    all_all_labels = []

    # 优化 1: 在进度条中显示当前 VLM 和 Task
    for batch in tqdm(dataloader, desc=f"Inferencing {vlm_type} on {task_name}"):
        # 准备输入数据
        embeddings = batch["hidden_states"]
        input_ids = batch["input_ids"]
        token_type_ids = batch["token_type_ids"]
        embeddings = embeddings.to(device)
        input_ids = input_ids.to(device)
        token_type_ids = token_type_ids.to(device)
        attention_masks = batch["attention_masks"].to(device)
      
        image_paths = batch["image_paths"]
        logits = batch["logits"]
        images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

        with torch.no_grad():
            # 设置 is_all=True 并获取 all_logits
            all_logits = model(
                input_ids=input_ids,
                embeddings=embeddings,
                logits=logits,  
                attention_masks=attention_masks,
                token_type_ids=token_type_ids,
                images=images,
                is_all=True,  
            )
            
            # 处理 attention mask，过滤掉 padding token
            attention_masks = attention_masks.view(-1)  # (B * L,)
            nonzero_indices = attention_masks.nonzero(as_tuple=True)

            # 展平 logits 并过滤
            all_logits = all_logits.view(-1, all_logits.size(-1))  # (B * L, C)
            filtered_all_logits = all_logits[nonzero_indices]
            
            # 计算概率
            filtered_all_probs = torch.softmax(filtered_all_logits, dim=-1)
            
            # 获取标签并过滤
            all_labels = batch["all_labels"].to(device)
            all_labels = all_labels.view(-1)  # (B * L)
            filtered_all_labels = all_labels[nonzero_indices]

        # 存储结果
        all_all_probs.append(filtered_all_probs.cpu())
        all_all_labels.append(filtered_all_labels.cpu())

    # 拼接所有 batch 的结果
    all_all_probs = torch.cat(all_all_probs).numpy()
    all_all_labels = torch.cat(all_all_labels).numpy()

    # 计算 AUROC 和 AUPR (基于连续概率值，无需阈值)
    logger.info("Calculate AUROC and AUPR")
    y_true = all_all_labels
    y_scores = all_all_probs[:, 1]  # 使用正类（如：幻觉）的预测概率
    
    # 修复 3: 防止数据集中只有单一类别导致 sklearn 报错崩溃
    if len(set(y_true)) > 1:
        auroc = roc_auc_score(y_true, y_scores)
        aupr = average_precision_score(y_true, y_scores)
    else:
        logger.warning("Only one class present in y_true. AUROC and AUPR are undefined.")
        auroc = 0.0
        aupr = 0.0

    # 计算最佳阈值及 Precision, Recall, F1
    logger.info("Calculate optimal thresholds")
    thresholds = [i * 0.001 for i in range(1000)]

    best_thresholds = {}
    best_f1_scores = {}
    precision_at_best_thresholds = {}
    recall_at_best_thresholds = {}

    task = 'all'
    probs = all_all_probs
    labels = all_all_labels
    
    best_f1 = 0
    best_threshold = 0
    precision_at_best_threshold = 0
    recall_at_best_threshold = 0
    
    # 优化 2: 在阈值计算进度条中显示当前 VLM 和 Task
    for threshold in tqdm(thresholds, desc=f"Calculating threshold for {vlm_type} on {task_name}"):
        # 预测：如果正类概率 > 阈值，则预测为正类 (1)
        preds = (probs[:, 1] > threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            precision_at_best_threshold = precision
            recall_at_best_threshold = recall
            
    best_thresholds[task] = best_threshold
    best_f1_scores[task] = best_f1
    precision_at_best_thresholds[task] = precision_at_best_threshold
    recall_at_best_thresholds[task] = recall_at_best_threshold

    # 汇总所有指标
    result = {
        "checkpoint_path": checkpoint_path,
        "threshold": best_thresholds,
        "precision": precision_at_best_thresholds,
        "recall": recall_at_best_thresholds,
        "f1_score": best_f1_scores,
        "auroc": {task: auroc},
        "aupr": {task: aupr},
    }

    logger.info(f"Saving results for {vlm_type} ({task_name}) to {save_filename}")
    logger.info(result)
    
    # 修复 4: 确保父目录存在，防止 FileNotFoundError
    parent_dir = os.path.dirname(save_filename)
    os.makedirs(parent_dir, exist_ok=True)
    
    with open(save_filename, "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    # 配置多个 ckpt 路径
    CKPT_PATHS = {
        "llava_mrg": "./checkpoint/hecp_llava.ckpt",
        "qwen2_5_mrg": "./checkpoint/hecp_qwen2_5.ckpt",
        "qwen_mrg": "./checkpoint/hecp_qwen3.ckpt",
       
    }
    
    # 配置需要遍历的 VLM 和 Task (验证集)
    VLMS = ["llava_mrg", "qwen2_5_mrg", "qwen_mrg", ]  # "qwen2_5_mrg", "llava_mrg", "qwen_mrg"
    TASKS = ["vqa", "instruct", "caption"]
    
    for vlm in VLMS:
        for task in TASKS:
            # 拼接数据目录和保存路径
            data_dir = f"/root/autodl-tmp/Halloc/ebdings/val/vlm_{task}/{vlm}"
            save_filename = f"/root/autodl-tmp/metrics/{vlm}/hecp/metric_{vlm}_{task}.json"
            checkpoint_path = CKPT_PATHS.get(vlm)
            
            # 调用评估函数，并传入 vlm_type 和 task_name
            calculate_optimal_threshold(
                data_dir=data_dir,
                save_filename=save_filename,
                checkpoint_path=checkpoint_path,
                device="cuda:0",
                vlm_type=vlm,
                task_name=task  
            )

"""
运行命令:
CUDA_VISIBLE_DEVICES=1 python -m scripts.evaluate_all.calculate_metrics_hecp

"""