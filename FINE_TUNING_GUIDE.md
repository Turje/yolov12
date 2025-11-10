# Fine-Tuning Guide: Private Dataset with Advanced Features

This guide explains how to use the enhanced training pipeline for fine-tuning on the private dataset with two-phase training, domain-wise evaluation, and per-class metrics.

## Overview

The pipeline now supports:
- **Two-phase training**: Freeze backbone initially, then unfreeze with scaled learning rate
- **Grouped splitting**: Keep augmented variants together to avoid data leakage
- **Sampling control**: Balance original vs augmented images in training
- **Domain-wise evaluation**: Evaluate separately on originals, augmented, and combined sets
- **Per-class metrics**: Track which classes perform best/worst

## Step 1: Create Grouped Splits with Subset Files

For the private dataset, use grouped splitting to ensure all variants of an image stay in the same split:

```bash
python scripts/split_dataset.py \
  --coco_json /content/datasets/private/merged_private.json \
  --images_dir /content/datasets/private \
  --output_dir /content/splits/private \
  --dataset_type private \
  --grouped \
  --original_ratio 0.6 \
  --emit_subsets \
  --seed 42
```

**Arguments:**
- `--grouped`: Groups all augmented variants (left/right rotations) with their original
- `--original_ratio 0.6`: Targets 60% originals, 40% augmented in training set
- `--emit_subsets`: Creates separate files for evaluation:
  - `train_originals.txt`, `train_augmented.txt`
  - `val_originals.txt`, `val_augmented.txt`
  - `test_originals.txt`, `test_augmented.txt`

**Output:**
```
/content/splits/private/
  ├── train.txt              # Balanced training set
  ├── train_originals.txt    # Only original images
  ├── train_augmented.txt    # Only augmented images
  ├── val.txt                # Full validation set
  ├── val_originals.txt
  ├── val_augmented.txt
  ├── test.txt
  ├── test_originals.txt
  ├── test_augmented.txt
  └── split_info.json
```

## Step 2: Fine-Tune with Two-Phase Training

Use the non-private best weights as initialization:

```bash
python scripts/train_private.py \
  --data_yaml /content/vizwiz/data/private.yaml \
  --checkpoint /content/yolov12_runs/nonprivate/checkpoints/best.pt \
  --project_dir /content/yolov12_runs/private \
  --wandb_project vizwiz_yolov12 \
  --model_size m \
  --img_size 800 \
  --epochs 100 \
  --freeze_epochs 5 \
  --lr_scale 0.1 \
  --patience 20 \
  --batch_size 8 \
  --eval_subsets /content/splits/private/val_originals.txt \
                 /content/splits/private/val_augmented.txt \
  --seed 42
```

**Key Arguments:**
- `--checkpoint`: Path to non-private best.pt (transfer learning)
- `--freeze_epochs 5`: Freeze backbone for first 5 epochs
- `--lr_scale 0.1`: Use 10% of base LR (0.001) for phase 2
- `--eval_subsets`: Optional list of subset files for domain-wise evaluation

**Training Process:**
1. **Phase 1 (epochs 1-5)**: Frozen backbone, only head trains
2. **Phase 2 (epochs 6-100)**: Unfrozen backbone, scaled LR (0.001), early stopping active

**W&B Logging:**
- Overall metrics: `final/map`, `final/map50`, `final/map75`
- Per-class AP: `per_class_ap/{class_name}` for each class
- Per-class AP table with top/bottom performers
- Domain-wise metrics (if `--eval_subsets` provided):
  - `val_originals/map`, `val_originals/map50`, ...
  - `val_augmented/map`, `val_augmented/map50`, ...

## Step 3: Standalone Domain-Wise Evaluation

After training, evaluate on multiple subsets independently:

```bash
python scripts/eval_subsets.py \
  --weights /content/yolov12_runs/private/checkpoints/best.pt \
  --subsets /content/splits/private/test_originals.txt \
           /content/splits/private/test_augmented.txt \
           /content/splits/private/test.txt \
  --data_yaml /content/vizwiz/data/private.yaml \
  --wandb_project vizwiz_yolov12 \
  --run_name private_test_eval \
  --img_size 800
```

**Output:**
- Separate metrics for each subset (e.g., `test_originals/map`, `test_augmented/map`)
- Per-class AP tables for each subset
- Comparison of performance across domains

## Understanding the Results

### Per-Class AP Analysis

Look at the per-class AP table in W&B to identify:
- **Top performers**: Classes the model detects well
- **Bottom performers**: Classes needing more data or better annotations

Example output:
```
=== Top 5 Classes by AP ===
1. credit_or_debit_card: 0.8542
2. condom_box: 0.7891
3. bank_statement: 0.7234
4. business_card: 0.6987
5. pregnancy_test: 0.6543

=== Bottom 5 Classes by AP ===
1. tattoo_sleeve: 0.2145
2. empty_pill_bottle: 0.2876
3. condom_with_plastic_bag: 0.3421
4. medical_record_document: 0.3987
5. transcript: 0.4123
```

### Domain-Wise Performance

Compare metrics across domains:
- If `val_originals/map > val_augmented/map`: Model struggles with zoomed/rotated views
- If `val_augmented/map > val_originals/map`: Model may be overfitting to augmentations
- Ideally, both should be similar

### Adjustments Based on Results

**If augmented performance is much lower:**
- Increase `--original_ratio` to 0.7 (more originals in training)
- Reduce on-the-fly augmentations (mosaic, mixup)
- Add more labeled augmented images

**If class performance is very imbalanced:**
- Consider class-weighted loss (requires code modification)
- Oversample rare classes in training data
- Review annotations for low-performing classes

**If overfitting occurs:**
- Increase `--patience` for earlier stopping
- Extend `--freeze_epochs` to 8-10
- Reduce `--lr_scale` to 0.05

## Example: Complete Workflow

```bash
# 1. Split with grouping and balancing
python scripts/split_dataset.py \
  --coco_json /content/datasets/private/merged_private.json \
  --images_dir /content/datasets/private \
  --output_dir /content/splits/private \
  --dataset_type private \
  --grouped \
  --original_ratio 0.6 \
  --emit_subsets

# 2. Fine-tune with two phases
python scripts/train_private.py \
  --data_yaml /content/vizwiz/data/private.yaml \
  --checkpoint /content/yolov12_runs/nonprivate/checkpoints/best.pt \
  --project_dir /content/yolov12_runs/private \
  --epochs 100 \
  --freeze_epochs 5 \
  --lr_scale 0.1 \
  --eval_subsets /content/splits/private/val_originals.txt \
                 /content/splits/private/val_augmented.txt

# 3. Evaluate on test sets
python scripts/eval_subsets.py \
  --weights /content/yolov12_runs/private/checkpoints/best.pt \
  --subsets /content/splits/private/test_originals.txt \
           /content/splits/private/test_augmented.txt \
           /content/splits/private/test.txt \
  --data_yaml /content/vizwiz/data/private.yaml \
  --run_name private_final_eval
```

## Tips

1. **Monitor W&B**: Watch the per-class AP tables to identify problem classes early
2. **Domain balance**: Keep validation sets mostly original to match deployment
3. **Learning rate**: If phase 2 diverges, reduce `--lr_scale` to 0.05
4. **Freezing**: For very different domains, extend `--freeze_epochs` to 10
5. **Checkpoints**: Always save to Google Drive for persistence across sessions

## Output Files

After training, you'll have:
```
/content/yolov12_runs/private/
  └── private_training/
      ├── weights/
      │   ├── best.pt          # Best validation checkpoint
      │   └── last.pt          # Latest checkpoint
      ├── results.png          # Training curves
      ├── confusion_matrix.png
      └── ...
  └── checkpoints/
      └── best.pt              # Copy for easy access
```

## W&B Dashboard

Check your W&B run for:
- Training curves (loss, mAP over epochs)
- Per-class AP table (sortable)
- Domain-wise comparison charts
- Confusion matrices
- Sample predictions

## Next Steps

After fine-tuning:
1. Review per-class AP to identify weak classes
2. Compare domain-wise performance (originals vs augmented)
3. If needed, collect more data for weak classes or domains
4. Use the best checkpoint for deployment/inference

