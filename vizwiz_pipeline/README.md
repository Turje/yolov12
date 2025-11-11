# VizWiz Few-Shot Object Detection Pipeline

Complete pipeline for training YOLOv12 on privacy-sensitive object detection using the VizWiz dataset, with few-shot learning protocol matching the BIV-Priv-Seg (WACV 2025) paper.

## Overview

This pipeline implements few-shot object detection for 16 privacy-sensitive categories (documents, medical items, etc.) using:
- **Base Training:** 100 non-private VizWiz classes
- **Few-Shot Fine-Tuning:** 1, 3, 5, 10 images per private class
- **Evaluation:** COCO-style mAP metrics with comparison to DeFRCN baseline

## Quick Start

### 1. Setup

```bash
# Install dependencies
pip install ultralytics wandb pycocotools

# Configure Weights & Biases
export WANDB_API_KEY=your_key_here
wandb login
```

### 2. Prepare Data

```bash
# Build private dataset
python scripts/build_private_coco.py \
    --query_json /path/to/query_images/instances_default.json \
    --shifted_json /path/to/augmented/instances_shifted_from_original.json \
    --left_rotate_dir /path/to/augmented/left_rotate \
    --right_rotate_dir /path/to/augmented/right_rotate \
    --output_json /path/to/merged_private.json

# Create dataset splits
python scripts/split_dataset.py \
    --coco_json /path/to/merged_private.json \
    --output_dir /path/to/splits \
    --train_ratio 0.8 \
    --val_ratio 0.1 \
    --test_ratio 0.1
```

### 3. Train Base Model (Non-Private)

```bash
python scripts/train_nonprivate.py \
    --data_yaml data/nonprivate.yaml \
    --project_dir /path/to/outputs/nonprivate \
    --epochs 250 \
    --batch_size 8 \
    --img_size 800
```

### 4. Few-Shot Training

```bash
# Create K-shot splits
python scripts/create_fewshot_splits.py \
    --input_json /path/to/merged_private.json \
    --output_dir /path/to/fewshot_splits \
    --k_shots 1 3 5 10 \
    --seed 42

# Train K-shot models
python scripts/train_fewshot.py \
    --k_shot 10 \
    --checkpoint /path/to/nonprivate/best.pt \
    --data_yaml /path/to/private_10shot.yaml \
    --project_dir /path/to/outputs/fewshot \
    --epochs 600 \
    --optimizer SGD \
    --protocol paper
```

### 5. Average Results Across Seeds

```bash
python scripts/average_seed_results.py \
    --protocol paper \
    --base_dir /path/to/outputs \
    --seeds 1 2 3 \
    --k_shots 1 3 5 10
```

## Training Protocols

### Paper Protocol (Match BIV-Priv-Seg)

- **Optimizer:** SGD
- **Epochs:** Adaptive (1500/1000/800/600 for 1/3/5/10-shot)
- **Augmentation:** Minimal (flip + brightness only)
- **Seeds:** 3-5 random seeds
- **Target:** Match/beat DeFRCN baseline

```bash
bash run_paper_protocol.sh
```

### Modern Protocol (Optimized YOLOv12)

- **Optimizer:** Adam
- **Epochs:** 100
- **Augmentation:** Heavy (mosaic, mixup, etc.)
- **Seeds:** 1 (seed=42)

```bash
bash run_modern_protocol.sh
```

## Directory Structure

```
vizwiz_pipeline/
├── data/
│   ├── nonprivate.yaml          # 100-class configuration
│   └── private.yaml             # 16-class configuration
├── scripts/
│   ├── build_private_coco.py    # Merge query + augmented annotations
│   ├── split_dataset.py         # Create train/val/test splits
│   ├── train_nonprivate.py      # Base model training
│   ├── create_fewshot_splits.py # Generate K-shot datasets
│   ├── train_fewshot.py         # Few-shot fine-tuning
│   ├── average_seed_results.py  # Aggregate results across seeds
│   └── wandb_setup.py           # Weights & Biases configuration
├── run_paper_protocol.sh        # Paper protocol training script
├── run_modern_protocol.sh       # Modern protocol training script
└── README.md                    # Documentation
```

## Expected Results

### Paper Protocol (Adaptive Epochs)

| K-shot | DeFRCN (baseline) | YOLOv12 (expected) | Training Time |
|--------|-------------------|-------------------|---------------|
| 1      | 18.7 mAP50       | 17-19 mAP50       | 3.5h/seed     |
| 3      | 26.2 mAP50       | 25-27 mAP50       | 5h/seed       |
| 5      | 30.1 mAP50       | 30-32 mAP50       | 6h/seed       |
| 10     | 35.4 mAP50       | 36-38 mAP50       | 7.5h/seed     |

Total training time: approximately 66 hours for 3 seeds

## Key Features

- **Grouped dataset splitting** - Prevents data leakage from augmented images
- **Isolated K-shot directories** - Bypasses Ultralytics auto-scanning issues
- **Weights & Biases integration** - Real-time training monitoring
- **COCO evaluation** - Standard mAP and size-based metrics (mAP-S/M/L)
- **Per-class AP** - Detailed performance breakdown per category
- **Multiple seeds** - Statistical robustness through seed averaging

## Citation

If you use this pipeline, please cite:

```bibtex
@inproceedings{tseng2025biv,
  title={BIV-Priv-Seg: Locating Private Content in Images Taken by People with Visual Impairments},
  author={Tseng, Fu-Jen and Gurari, Danna},
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  year={2025}
}
```

## Documentation

- **READY_TO_START.md** - Quick start guide with Google Colab setup
- **EPOCH_COMPARISON.md** - Analysis of training duration and iteration count
- **TWO_PROTOCOL_GUIDE.md** - Detailed comparison of training protocols

## Troubleshooting

### No labels found during training

```bash
# Regenerate YOLO labels from COCO JSON
python scripts/convert_coco_to_yolo.py --coco_json /path/to/merged_private.json

# Delete cached files
find /path/to/datasets -name "*.cache" -delete
```

### Weights & Biases not logging

```bash
# Ensure W&B is properly configured
export WANDB_API_KEY=your_key
wandb login

# Verify connection in Python
import wandb
print(wandb.Api().viewer.username)
```

### GPU out of memory

```bash
# Reduce batch size
--batch_size 2

# Reduce image size
--img_size 640
```

### Training stops unexpectedly

- Check Google Colab runtime status
- Verify sufficient storage in Google Drive
- Monitor GPU utilization with `nvidia-smi`

## Requirements

- Python 3.8+
- PyTorch 1.13+
- CUDA 11.3+ (for GPU training)
- 16GB+ GPU memory (recommended)
- Google Colab Pro (for extended runtimes)

## License

This project is licensed under the AGPL-3.0 License. See the LICENSE file for details.

## Contact

For questions or issues, please open an issue on the GitHub repository.

## Acknowledgments

- YOLOv12 implementation based on Ultralytics
- BIV-Priv-Seg dataset and protocol from Tseng & Gurari (2025)
- DeFRCN baseline comparison from the few-shot detection literature
