# Quick Reference: Private Fine-Tuning Commands

## Split Dataset (with grouping & balancing)

```bash
python scripts/split_dataset.py \
  --coco_json /content/datasets/private/merged_private.json \
  --images_dir /content/datasets/private \
  --output_dir /content/splits/private \
  --dataset_type private \
  --grouped \
  --original_ratio 0.6 \
  --emit_subsets
```

**Key flags:**
- `--grouped`: Keep augmented variants with originals (no leakage)
- `--original_ratio 0.6`: 60% originals, 40% augmented in training
- `--emit_subsets`: Create separate eval lists

## Fine-Tune (two-phase training)

```bash
python scripts/train_private.py \
  --data_yaml /content/vizwiz/data/private.yaml \
  --checkpoint /content/yolov12_runs/nonprivate/checkpoints/best.pt \
  --project_dir /content/yolov12_runs/private \
  --epochs 100 \
  --freeze_epochs 5 \
  --lr_scale 0.1 \
  --patience 20 \
  --eval_subsets /content/splits/private/val_originals.txt \
                 /content/splits/private/val_augmented.txt
```

**Key flags:**
- `--checkpoint`: Initialize from non-private weights
- `--freeze_epochs 5`: Freeze backbone first 5 epochs
- `--lr_scale 0.1`: Use LR=0.001 in phase 2 (10% of 0.01)
- `--eval_subsets`: Optional domain-wise eval

## Evaluate on Test Sets

```bash
python scripts/eval_subsets.py \
  --weights /content/yolov12_runs/private/checkpoints/best.pt \
  --subsets /content/splits/private/test_originals.txt \
           /content/splits/private/test_augmented.txt \
           /content/splits/private/test.txt \
  --data_yaml /content/vizwiz/data/private.yaml \
  --run_name private_test_eval
```

## What Gets Logged to W&B

### During Training
- `final/map`, `final/map50`, `final/map75` (combined validation)
- `per_class_ap/{class_name}` for each class
- `per_class_ap_table` (sortable table)
- Top 5 & Bottom 5 classes printed
- `val_originals/map`, `val_augmented/map` (if `--eval_subsets` used)

### During Evaluation
- `{subset_name}/map`, `{subset_name}/map50`, `{subset_name}/map75`
- `{subset_name}_per_class_ap_table` for each subset
- Top 3 & Bottom 3 classes per subset

## Adjusting Parameters

### If augmented performance is weak:
```bash
--original_ratio 0.7    # More originals
--freeze_epochs 8       # Longer freeze
```

### If overfitting:
```bash
--patience 15           # Earlier stopping
--lr_scale 0.05         # Lower LR
```

### For small dataset or very different domain:
```bash
--freeze_epochs 10      # Longer freeze
--lr_scale 0.05         # Lower LR
--epochs 150            # More epochs
```

## File Structure After Training

```
/content/yolov12_runs/private/
  └── private_training/
      ├── weights/
      │   ├── best.pt          ← Use for inference
      │   └── last.pt          ← Resume training if needed
      └── checkpoints/
          └── best.pt          ← Copy for easy access
```

## Tips

1. **Always use `--grouped`** for private dataset (prevents leakage)
2. **Monitor W&B** for per-class AP during training
3. **Save to Drive** so checkpoints persist across sessions
4. **Test on multiple domains** to ensure robustness
5. **Check top/bottom classes** to identify data quality issues

