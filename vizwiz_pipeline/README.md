# VizWiz YOLOv12 Training Pipeline

This folder contains the complete training pipeline for YOLOv12 on the VizWiz dataset.

## Structure

```
vizwiz_pipeline/
├── scripts/              # Training and evaluation scripts
│   ├── build_private_coco.py
│   ├── convert_coco_to_yolo.py
│   ├── evaluate_private.py
│   ├── split_dataset.py
│   ├── train_nonprivate.py
│   ├── train_private.py
│   └── wandb_setup.py
├── data/                 # Dataset configurations
│   ├── nonprivate.yaml
│   └── private.yaml
├── README_pipeline.md    # Complete pipeline documentation
└── BRANCH_ORGANIZATION.md # Branch organization guide
```

## Quick Start

See `README_pipeline.md` for complete instructions.

## Usage

All scripts should be run from the repository root or this folder. Paths in the scripts are configured for Google Colab with Google Drive mounted at `/content/drive/MyDrive/`.

