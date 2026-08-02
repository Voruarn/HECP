# HECP: Heterogeneous Evidence Collaboration Probe for Token-Level Hallucination Localization in Large Vision Language Models

Official codebase for the paper:
**"Heterogeneous Evidence Collaboration for Token-Level Hallucination Localization in Large Vision Language Models"**

---

## Overview

- We identify the fundamental limitation of existing token-level hallucination detection methods: the reliance on unimodal evidence, and propose the principle of heterogeneous evidence collaboration as a more effective paradigm for hallucination localization in LVLMs.
- We propose HECP, a novel framework that integrates internal hidden state dynamics (via HSDA) with external text-visual grounding verification (via TIG), fused through CSFT and refined by MSTA, enabling comprehensive and complementary hallucination evidence capture.
- Extensive experiments across three mainstream LVLMs demonstrate that HECP achieves state-of-the-art performance, with significant improvements over existing baselines. Ablation studies confirm the necessity of each module and the complementarity of heterogeneous signals.

Our HECP operates on multi-stage VLM hidden states + VLM Response Text + images.

---

## Repository Structure

```
dataset_halloc/
├── train.py                               # Training entry point (Hydra + PyTorch Lightning)
├── config/                                # Hydra configuration files
│   ├── train.yaml                         #   root config
│   ├── datamodule/                        #   data loading configs
│   ├── dataset/                           #   dataset configs
│   ├── model/                             #   model architecture configs
│   ├── module/                            #   lightning module configs
│   ├── optimizer/                         #   optimizer configs (AdamW)
│   ├── scheduler/                         #   LR scheduler configs (cosine, multi-step)
│   ├── loss/                              #   loss function configs (cross-entropy)
│   ├── metric/                            #   metric configs (classification)
│   ├── callback/                          #   callback configs
│   ├── trainer/                           #   trainer configs
│   └── experiment/                        #   experiment presets
│
├── src/                                   # Core library
│   ├── model/                             #   Model definitions
│   │   ├── halloc.py                      #     halLocalizer-VLM: embedding-based model (VLM hidden states)
│   │   └── hecp.py                        #     Our HECP
│   ├── module/                            #   PyTorch Lightning modules
│   │   ├── loss/                          #     loss functions
│   │   ├── metric/                        #     evaluation metrics
│   │   └── optimizer/                     #     optimizer + scheduler setup
│   ├── datamodule/                        #   data modules + datasets
│   │   └── dataset/                       #     dataset implementations
│   ├── message/                           #   inter-component messaging
│   └── utils/                             #   logging utilities
│
└── scripts/                               # Pipeline scripts
    ├── extract/                           #   Step 1: Extract VLM embeddings and Response Text
    │   ├── hecp_extract_vlm_ebdings_llava.py
    │   ├── hecp_extract_vlm_ebdings_qwen2_5.py
    │   ├── hecp_extract_vlm_ebdings_qwen3.py
    │   ├── merge_chunks_npy.py            #   if save_interval <100%, merge_chunks after extraction
    │   ├── extract_vlm_embeddings_text.py
    │   └── extract_vlm_embeddings_text_blur.py
    │
    ├── postprocess/                       #   Step 2: Align token indices
    │   ├── postprocess_llava.py
    │   ├── postprocess_qwen2_5.py
    │   ├── postprocess_qwen3.py
    │   ├── postprocess_minigpt4.py
    │   └── postprocess_text.py
    │
    ├── evaluate_all/                          #   Step 4: Evaluate + find thresholds, get hallucination probabilities
    │   ├── calculate_metrics_hecp.py
    │   └── get_prob_hecp.py
    │
    └── calibration/                       #   Step 5: Calibration analysis
        ├── calculate_calibration_error_halloc_ece.py
        ├── calculate_calibration_error_halloc_ace.py
        ├── calculate_calibration_error_internvl_ece.py
        ├── calculate_calibration_error_internvl_ace.py
        └── calculate_logprob_internvl.py
```

---

## Requirements

- Python >= 3.8
- PyTorch >= 1.13
- PyTorch Lightning
- Hydra (`hydra-core`, `hydra-colorlog`)
- HuggingFace Transformers
- Accelerate
- `fire`, `loguru`, `scikit-learn`, `Pillow`

For VLM-specific extraction scripts, you also need the corresponding VLM libraries:
- **LLaVA**: [liuhaotian/llava-v1.5-7b](https://github.com/haotian-liu/LLaVA)
- **Qwen2.5-VL-7B**: https://github.com/QwenLM/Qwen3-VL, respective repositories, or huggingface
- **Qwen3-VL-8B**: https://github.com/QwenLM/Qwen3-VL, respective repositories, or huggingface

---

## Dataset

- **HalLoc**: https://github.com/dbsltm/cvpr25_halloc
- **M-HalDetect**: https://github.com/hendryx-scale/mhal-detect

---

## Pipeline

The full workflow has five stages:

### Step 1: Postprocess annotations

Align character-level hallucination annotations to VLM-specific tokenizations:

```bash
python scripts/postprocess/postprocess_llava.py \
    --input_path data/halloc_train.json
```

### Step 2: Extract VLM embeddings and Response Text

Run a VLM in forward mode to extract hidden-state embeddings for each sample. Uses HuggingFace Accelerate for multi-GPU:

```bash
accelerate launch scripts/extract/hecp_extract_vlm_ebdings_llava.py \
    --data_path data/halloc_train_llava_postprocessed.json \
    --image_dir /path/to/images \
    --save_dir data/train/vlm_embeddings/llava \
    --batch_size 8
```

### Step 3: Train HECP

Train the hallucination detection model using Hydra configs:

```bash
python train.py \
    experiment.name=hecp_llava \
    experiment.work_dir=./outputs
```


Override any config value via the command line (Hydra syntax). See `config/` for all options.

### Step 4: Evaluate

Find optimal per-category thresholds on validation set, then evaluate:

```bash
python scripts/evaluate_all/calculate_metrics_hecp.py \
    --checkpoint_path outputs/halloc_llava/best.ckpt \
    --data_dir data/val/vlm_embeddings/llava

python scripts/evaluate/evaluate_single.py \
    --threshold_path evaluation/thresholds/thresholds.json \
    --checkpoint_path outputs/halloc_llava/best.ckpt \
    --data_dir data/val/vlm_embeddings/llava
```

### Step 5: Calibration analysis (optional)

Compute Expected Calibration Error (ECE) and Adaptive Calibration Error (ACE) with temperature scaling:

```bash
python scripts/calibration/calculate_calibration_error_halloc_ece.py --subset vqa
python scripts/calibration/calculate_calibration_error_internvl_ece.py --subset vqa
```
---

## Acknowledgements

This repo is built upon HalLoc (https://github.com/dbsltm/cvpr25_halloc).

## Citation

If you find this work helpful, please consider citing:

The paper is still undergoing review.


