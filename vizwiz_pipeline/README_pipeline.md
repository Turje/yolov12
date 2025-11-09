# YOLOv12 Training Pipeline for VizWiz Dataset

This repository contains a complete two-stage training pipeline for YOLOv12 on the VizWiz dataset:
1. **Stage 1**: Train on non-private base images (100 categories)
2. **Stage 2**: Fine-tune on private query images with augmentations (16 categories)

## Overview

The pipeline is organized into branches and scripts:

- **Branches**: `train_nonprivate`, `train_private`, `evaluate`, `utils_wandb`
- **Scripts**: Dataset preparation, training, and evaluation scripts
- **Data Configs**: YAML files for dataset configuration

## Dataset Structure

### Non-Private Dataset
- **Location**: `/content/drive/MyDrive/datasets/nonprivate/base_images/`
- **Annotations**: `instances_default.json` (COCO format)
- **Images**: ~4000 images
- **Categories**: 100 non-private object categories

### Private Dataset
- **Location**: `/content/drive/MyDrive/datasets/private/`
- **Folders**:
  - `query_images/` - Original query images
  - `left_rotate/` - Left-rotated augmented images
  - `right_rotate/` - Right-rotated augmented images
- **Annotations**:
  - `query_images/instances_default.json`
  - `left_rotate/instances_shifted_from_original.json`
  - `right_rotate/instances_shifted_from_original.json`
- **Images**: ~5000 images total
- **Categories**: 16 private object categories

## Setup

### 1. Mount Google Drive in Colab

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 2. Install Dependencies

```bash
pip install ultralytics pycocotools wandb
```

### 3. Set Up Weights & Biases

```bash
export WANDB_API_KEY=your_api_key_here
wandb login
```

Or set it in your Colab environment:
```python
import os
os.environ["WANDB_API_KEY"] = "your_api_key_here"
```

### 4. Clone/Setup Repository

```bash
# If using the YOLOv12 repo
git clone https://github.com/Turje/yolov12.git
cd yolov12

# Or if working in this repository
cd /content/drive/MyDrive/vizwiz
```

## Pipeline Steps

### Step 1: Merge Private Dataset Annotations

Merge the private dataset annotations from query_images, left_rotate, and right_rotate into a single COCO file:

```bash
python scripts/build_private_coco.py \
    --query_json /content/drive/MyDrive/datasets/private/query_images/instances_default.json \
    --query_images_dir /content/drive/MyDrive/datasets/private/query_images \
    --left_rotate_dir /content/drive/MyDrive/datasets/private/left_rotate \
    --right_rotate_dir /content/drive/MyDrive/datasets/private/right_rotate \
    --shifted_json /content/drive/MyDrive/datasets/private/left_rotate/instances_shifted_from_original.json \
    --output_json /content/drive/MyDrive/datasets/private/merged_private.json
```

This creates a merged COCO annotation file that will be used for all private dataset training and evaluation.

### Step 2: Split Datasets

Split both datasets into 80/10/10 train/val/test splits:

#### Non-Private Split

```bash
python scripts/split_dataset.py \
    --coco_json /content/drive/MyDrive/datasets/nonprivate/base_images/instances_default.json \
    --images_dir /content/drive/MyDrive/datasets/nonprivate/base_images \
    --output_dir /content/splits/nonprivate \
    --dataset_type nonprivate
```

#### Private Split

```bash
python scripts/split_dataset.py \
    --coco_json /content/drive/MyDrive/datasets/private/merged_private.json \
    --images_dir /content/drive/MyDrive/datasets/private \
    --output_dir /content/splits/private \
    --dataset_type private
```

This creates:
- `train.txt`, `val.txt`, `test.txt` files listing image paths
- `split_info.json` with image IDs for each split

**Note**: The test splits (10% each) are reserved for final evaluation and are not used during training.

### Step 2.5: Convert COCO to YOLO Format

YOLOv12 requires YOLO format (images + .txt label files). Convert each split:

#### Non-Private Conversion

```bash
# Convert train split
python scripts/convert_coco_to_yolo.py \
    --coco_json /content/drive/MyDrive/datasets/nonprivate/base_images/instances_default.json \
    --split_file /content/splits/nonprivate/train.txt \
    --output_dir /content/drive/MyDrive/datasets/nonprivate/yolo_format/train \
    --split_name train

# Convert val split
python scripts/convert_coco_to_yolo.py \
    --coco_json /content/drive/MyDrive/datasets/nonprivate/base_images/instances_default.json \
    --split_file /content/splits/nonprivate/val.txt \
    --output_dir /content/drive/MyDrive/datasets/nonprivate/yolo_format/val \
    --split_name val

# Convert test split
python scripts/convert_coco_to_yolo.py \
    --coco_json /content/drive/MyDrive/datasets/nonprivate/base_images/instances_default.json \
    --split_file /content/splits/nonprivate/test.txt \
    --output_dir /content/drive/MyDrive/datasets/nonprivate/yolo_format/test \
    --split_name test
```

#### Private Conversion

```bash
# Convert train split
python scripts/convert_coco_to_yolo.py \
    --coco_json /content/drive/MyDrive/datasets/private/merged_private.json \
    --split_file /content/splits/private/train.txt \
    --output_dir /content/drive/MyDrive/datasets/private/yolo_format/train \
    --split_name train

# Convert val split
python scripts/convert_coco_to_yolo.py \
    --coco_json /content/drive/MyDrive/datasets/private/merged_private.json \
    --split_file /content/splits/private/val.txt \
    --output_dir /content/drive/MyDrive/datasets/private/yolo_format/val \
    --split_name val

# Convert test split
python scripts/convert_coco_to_yolo.py \
    --coco_json /content/drive/MyDrive/datasets/private/merged_private.json \
    --split_file /content/splits/private/test.txt \
    --output_dir /content/drive/MyDrive/datasets/private/yolo_format/test \
    --split_name test
```

This creates YOLO format directories with:
- `images/` - Image files
- `labels/` - Corresponding .txt label files

### Step 3: Update Dataset YAMLs

Update the dataset YAML files with correct paths and class names:

#### `data/nonprivate.yaml`

```yaml
path: /content/drive/MyDrive/datasets/nonprivate/yolo_format
train: train/images
val: val/images
test: test/images

nc: 100
names:
  0: class_0
  1: class_1
  # ... add all 100 class names from your COCO annotations
```

#### `data/private.yaml`

```yaml
path: /content/drive/MyDrive/datasets/private/yolo_format
train: train/images
val: val/images
test: test/images

nc: 16
names:
  0: local_newspaper
  1: bank_statement
  # ... (already configured with 16 classes)
```

**Note**: After running the conversion script, you can extract the class names from the COCO JSON file to populate the `names` field in the YAML files.

### Step 4: Train on Non-Private Dataset

Train YOLOv12 on the non-private base images:

```bash
python scripts/train_nonprivate.py \
    --data_yaml data/nonprivate.yaml \
    --project_dir /content/drive/MyDrive/yolov12_runs/nonprivate \
    --wandb_project vizwiz_yolov12 \
    --model_size m \
    --img_size 800 \
    --epochs 250 \
    --patience 50 \
    --batch_size 8
```

**Training Configuration**:
- Model size: `m` (Medium)
- Image size: `800`
- Max epochs: `250`
- Early stopping patience: `50`
- Batch size: `8`

**Outputs**:
- Checkpoints saved to: `/content/drive/MyDrive/yolov12_runs/nonprivate/checkpoints/best.pt`
- Training logs and metrics logged to W&B project: `vizwiz_yolov12`
- Best model saved as W&B artifact: `nonprivate_best_model`

### Step 5: Fine-tune on Private Dataset

Fine-tune the model on the private dataset, starting from the non-private checkpoint:

```bash
python scripts/train_private.py \
    --data_yaml data/private.yaml \
    --checkpoint /content/drive/MyDrive/yolov12_runs/nonprivate/checkpoints/best.pt \
    --project_dir /content/drive/MyDrive/yolov12_runs/private \
    --wandb_project vizwiz_yolov12 \
    --model_size m \
    --img_size 800 \
    --epochs 250 \
    --patience 50 \
    --batch_size 8
```

**Outputs**:
- Checkpoints saved to: `/content/drive/MyDrive/yolov12_runs/private/checkpoints/best.pt`
- Training logs and metrics logged to W&B project: `vizwiz_yolov12`
- Best model saved as W&B artifact: `private_best_model`

### Step 6: Evaluate on Test Sets

Evaluate the final model on both test sets (non-private and private):

```bash
python scripts/evaluate_private.py \
    --checkpoint /content/drive/MyDrive/yolov12_runs/private/checkpoints/best.pt \
    --nonprivate_yaml data/nonprivate.yaml \
    --private_yaml data/private.yaml \
    --output_dir /content/drive/MyDrive/yolov12_runs/eval
```

**Metrics Tracked**:
- `mAP` (mean Average Precision)
- `mAP50` (mAP at IoU=0.50)
- `mAP75` (mAP at IoU=0.75)
- `mAP50:95` (mAP averaged over IoU thresholds 0.50-0.95)
- `mAP_small` (mAP for small objects)
- `mAP_medium` (mAP for medium objects)
- `mAP_large` (mAP for large objects)

**Outputs**:
- JSON results: `/content/drive/MyDrive/yolov12_runs/eval/evaluation_results_TIMESTAMP.json`
- CSV results: `/content/drive/MyDrive/yolov12_runs/eval/evaluation_results_TIMESTAMP.csv`

## Weights & Biases Integration

### Project Setup

All training runs are logged to the W&B project: `vizwiz_yolov12`

### Artifacts

Model checkpoints are automatically saved as W&B artifacts:
- `nonprivate_best_model`: Best model from non-private training
- `private_best_model`: Best model from private fine-tuning

You can download these artifacts later from the W&B web interface.

### Metrics Logged

During training, the following metrics are logged:
- Loss components (box, cls, dfl)
- Validation metrics (mAP, mAP50, mAP75, mAP50:95)
- Size-specific mAPs (small, medium, large)
- Learning rate
- Epoch progress

## File Structure

```
vizwiz/
├── data/
│   ├── nonprivate.yaml      # Non-private dataset config
│   └── private.yaml         # Private dataset config
├── scripts/
│   ├── build_private_coco.py    # Merge private annotations
│   ├── split_dataset.py          # Create train/val/test splits
│   ├── train_nonprivate.py       # Stage 1 training
│   ├── train_private.py          # Stage 2 fine-tuning
│   ├── evaluate_private.py       # Final evaluation
│   └── wandb_setup.py            # W&B helper functions
├── README_pipeline.md            # This file
└── .gitignore                    # Git ignore rules
```

## Important Notes

1. **Data Not in Repo**: The actual dataset images and annotations are stored in Google Drive, not in this repository. Only configuration files and scripts are tracked.

2. **Path Configuration**: All paths in the scripts use `/content/drive/MyDrive/...` for Colab. Adjust these paths if using a different setup.

3. **Test Sets Reserved**: The 10% test splits for both datasets are reserved for final evaluation and are not used during training or validation.

4. **Checkpoint Saving**: Checkpoints are saved both to Google Drive (for easy access) and as W&B artifacts (for versioning and sharing).

5. **Early Stopping**: Training will automatically stop if validation mAP doesn't improve for 50 epochs (patience=50).

6. **Model Size**: The pipeline uses YOLOv12-m (medium) by default. You can change this with the `--model_size` argument (n, s, m, l, x).

## Troubleshooting

### Issue: W&B API Key Not Found

**Solution**: Set the environment variable:
```bash
export WANDB_API_KEY=your_key_here
```

Or in Colab:
```python
import os
os.environ["WANDB_API_KEY"] = "your_key_here"
```

### Issue: Dataset Paths Not Found

**Solution**: Verify that:
1. Google Drive is mounted at `/content/drive`
2. Dataset files are in the expected locations
3. Paths in YAML files match actual directory structure

### Issue: Out of Memory

**Solution**: Reduce batch size:
```bash
--batch_size 4  # or lower
```

Or reduce image size:
```bash
--img_size 640  # instead of 800
```

### Issue: YOLOv12 Not Available

**Solution**: The scripts automatically fall back to YOLOv11 or YOLOv10 if YOLOv12 is not available. Make sure you have the latest `ultralytics` package:
```bash
pip install --upgrade ultralytics
```

## Citation

If you use this pipeline, please cite:
- YOLOv12: [Citation when available]
- VizWiz Dataset: [Citation when available]

## License

This pipeline follows the license of the YOLOv12 repository and VizWiz dataset.

