import os
import numpy as np
import torch
from tqdm import tqdm
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader

from src.model import get_model
from src.datamodule.dataset import get_dataset


def save_positive_probs(
    data_dir: str,
    save_npy_path: str,
    checkpoint_path: str = None,
    device: str = "cuda:0",
    vlm_type: str = "llava_mrg",
    task_name: str = "all",
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
        vlm_out_dim = 3584  # 修复 2: 修正 Qwen2.5 的隐藏层维度拼写错误 (3594 -> 3584)
        
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
    
    # 用于保存每个样本每个 token 的正类概率
    all_pos_probs_per_sample = []

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
            
            # 计算每个样本每个 token 的正类概率, 形状 [B, L]
            # all_logits: (B, L, C) -> softmax -> 取正类 (索引 1) 概率
            batch_pos_probs = torch.softmax(all_logits, dim=-1)[..., 1]  # (B, L)
            all_pos_probs_per_sample.append(batch_pos_probs.cpu().numpy())

    # ============ 直接拼接并保存为 npy ============
    # 因为所有输入数据的 token 长度已经统一，直接在第 0 维 (batch 维) 拼接即可
    pos_probs_array = np.concatenate(all_pos_probs_per_sample, axis=0)

    # 保存 npy 文件
    parent_dir = os.path.dirname(save_npy_path)
    os.makedirs(parent_dir, exist_ok=True)
    np.save(save_npy_path, pos_probs_array)
    logger.info(f"Saved positive-class probs to {save_npy_path}, shape={pos_probs_array.shape}")


if __name__ == "__main__":
    CKPT_PATHS = {
        "llava_mrg": "./checkpoint/hecp_llava.ckpt",
        "qwen2_5_mrg": "./checkpoint/hecp_qwen2_5.ckpt",
        "qwen_mrg": "./checkpoint/hecp_qwen3.ckpt",
    }
    
    VLMS = ["llava_mrg", "qwen2_5_mrg", "qwen_mrg", ]  # "qwen2_5_mrg", "llava_mrg", "qwen_mrg"
    TASKS = ["vqa", "instruct", "caption"]
    
    for vlm in VLMS:
        for task in TASKS:
            data_dir = f"/root/autodl-tmp/Halloc/ebdings/val/vlm_{task}/{vlm}"
            save_npy_path = f"/root/autodl-tmp/halloc_probs/{vlm}/hecp/probs_{vlm}_{task}.npy"
            checkpoint_path = CKPT_PATHS.get(vlm)
            
            save_positive_probs(
                data_dir=data_dir,
                save_npy_path=save_npy_path,
                checkpoint_path=checkpoint_path,
                device="cuda:0",
                vlm_type=vlm,
                task_name=task  
            )
    
"""
运行命令:
CUDA_VISIBLE_DEVICES=1 python -m scripts.evaluate_all.get_prob_hecp


"""