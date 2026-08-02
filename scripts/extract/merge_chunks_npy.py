import os
import re
import glob
import numpy as np
from tqdm import tqdm


def natural_sort_key(filepath):
    """按文件名中的数字自然排序，确保 chunk_0, chunk_1, ..., chunk_10 顺序正确"""
    filename = os.path.basename(filepath)
    return [int(c) if c.isdigit() else c for c in re.split(r'(\d+)', filename)]


def pad_to_max(arrays, pad_dims):
    """
    将一组数组在指定维度上 padding 到全局最大值。

    Args:
        arrays:   list of numpy arrays
        pad_dims: list of int, 需要 padding 的维度索引列表
                  例如 [1] 表示只 pad 第 1 维, [3, 4] 表示 pad 第 3 和第 4 维
    Returns:
        list of padded numpy arrays
    """
    if not pad_dims:
        return arrays

    # 计算每个需要 pad 的维度的全局最大值
    max_sizes = {}
    for dim in pad_dims:
        max_sizes[dim] = max(a.shape[dim] for a in arrays)

    padded = []
    for a in arrays:
        # 计算每个维度需要 pad 多少
        pad_widths = []
        needs_pad = False
        for dim in range(a.ndim):
            if dim in max_sizes and a.shape[dim] < max_sizes[dim]:
                pad_widths.append((0, max_sizes[dim] - a.shape[dim]))
                needs_pad = True
            else:
                pad_widths.append((0, 0))
        if needs_pad:
            a = np.pad(a, pad_widths, mode='constant', constant_values=0)
        padded.append(a)
    return padded


def merge_chunks(save_dir: str, output_dir: str = None):
    """
    将 save_dir 下所有 chunk 文件按类型分别合并为单独的文件。
    会自动处理不同 chunk 之间序列长度不一致的问题。

    Args:
        save_dir:   存放 chunk 文件的目录
        output_dir: 合并后文件的输出目录，默认与 save_dir 相同
    """
    if output_dir is None:
        output_dir = save_dir
    os.makedirs(output_dir, exist_ok=True)

    # ---- 定义所有需要合并的文件模式 ----
    # (基础文件名, 扩展名, 是否为 npz, 需要 pad 的维度列表)
    #
    # 各特征的 shape 说明：
    #   attention_masks:               (N, seq_len)                             → pad dim 1
    #   hidden_states_selected_layers: (N, num_layers, seq_len, hidden_dim)     → pad dim 2
    #   indices / ids / image_paths:   (N,)                                     → 无需 pad
    #   *_labels / label_masks:        (N, seq_len)                             → pad dim 1
    #
    file_specs = [
        ("attention_masks",               ".npy",  False, [1]),
        ("hidden_states_selected_layers", ".npy",  True,  [2]),
        ("indices",                       ".npy",  False, []),
        ("ids",                           ".npy",  False, []),
        ("image_paths",                   ".npy",  False, []),
        ("obj_labels",                    ".npy",  False, [1]),
        ("att_labels",                    ".npy",  False, [1]),
        ("rel_labels",                    ".npy",  False, [1]),
        ("sce_labels",                    ".npy",  False, [1]),
        ("oth_labels",                    ".npy",  False, [1]),
        ("all_labels",                    ".npy",  False, [1]),
        ("label_masks",                   ".npy",  False, [1]),
        ("input_ids",                     ".npy",  False, [1]),
        ("token_type_ids",                ".npy",  False, [1]),
    ]

    print(f"[INFO] Scanning chunk files in: {save_dir}")
    print(f"[INFO] Merged files will be saved to: {output_dir}")
    print("=" * 60)

    for base_name, ext, is_npz, pad_dims in file_specs:
        pattern = os.path.join(save_dir, f"{base_name}_chunk_*{ext}")
        chunk_files = sorted(glob.glob(pattern), key=natural_sort_key)

        if not chunk_files:
            print(f"[SKIP] No chunk files found for: {base_name}")
            continue

        print(f"\n[MERGE] {base_name}  ({len(chunk_files)} chunks)")
        for f in chunk_files:
            print(f"        {os.path.basename(f)}")

        # ---- 加载所有 chunk ----
        arrays = []
        for f in tqdm(chunk_files, desc=f"  Loading {base_name}", ncols=80):
            if is_npz:
                data = np.load(f)
                arr = data[list(data.keys())[0]]
                arrays.append(arr)
            else:
                arrays.append(np.load(f))

        # 打印各 chunk 的 shape 以便排查
        for i, a in enumerate(arrays):
            print(f"  chunk_{i} shape: {a.shape}")

        # ---- 对不同 seq_len 的维度进行 padding 对齐 ----
        if pad_dims:
            print(f"  Padding dims {pad_dims} to global max...")
            arrays = pad_to_max(arrays, pad_dims)

        # ---- 沿第 0 维（样本维度）拼接 ----
        merged = np.concatenate(arrays, axis=0)
        print(f"  Merged shape: {merged.shape}, dtype: {merged.dtype}")

        # ---- 保存合并后的文件 ----
        output_path = os.path.join(output_dir, f"{base_name}{ext}")
        if is_npz:
            np.savez_compressed(output_path, merged.astype(np.float16))
        else:
            np.save(output_path, merged)

        print(f"  Saved -> {output_path}")

        # 释放内存
        del arrays, merged

    print("\n" + "=" * 60)
    print("[DONE] All chunks merged successfully!")


if __name__ == "__main__":
    print('start merge!')
    
    MODES = ['train', 'val', ] 
    TASKS = ['instruct', 'vqa', 'caption']
    
    for mode in MODES:
        for task in TASKS:
            save_dir = f"/root/autodl-tmp/Halloc/ebdings/{mode}/vlm_{task}/qwen2_5"
            output_dir = f"/root/autodl-tmp/Halloc/ebdings/{mode}/vlm_{task}/qwen2_5_mrg"
            merge_chunks(save_dir, output_dir)
            print('done!!!')
    
    
