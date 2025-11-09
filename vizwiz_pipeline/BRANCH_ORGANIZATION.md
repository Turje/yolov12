# Branch Organization

This document explains which files belong to each branch.

## Main Branch
Contains:
- All utility scripts (shared across branches):
  - `scripts/split_dataset.py`
  - `scripts/build_private_coco.py`
  - `scripts/convert_coco_to_yolo.py`
- Configuration files:
  - `data/nonprivate.yaml`
  - `data/private.yaml`
- Documentation:
  - `README_pipeline.md`
  - `.gitignore`

## train_nonprivate Branch
Contains (in addition to main):
- `scripts/train_nonprivate.py` - Training script for non-private dataset
- Focus: Stage 1 training on base images

## train_private Branch
Contains (in addition to main):
- `scripts/train_private.py` - Fine-tuning script for private dataset
- Focus: Stage 2 fine-tuning on private images

## evaluate Branch
Contains (in addition to main):
- `scripts/evaluate_private.py` - Evaluation script for test sets
- Focus: Final evaluation on both datasets

## utils_wandb Branch
Contains (in addition to main):
- `scripts/wandb_setup.py` - W&B helper functions
- Focus: Weights & Biases integration utilities

## How to Organize Files on Branches

### Option 1: Keep all scripts on main, branches add specific training scripts
This is the recommended approach since all scripts work together.

### Option 2: Each branch has its own copy of relevant scripts
If you prefer to have isolated branches:

```bash
# On train_nonprivate branch
git checkout train_nonprivate
git add scripts/train_nonprivate.py
git commit -m "Add non-private training script"

# On train_private branch
git checkout train_private
git add scripts/train_private.py
git commit -m "Add private fine-tuning script"

# On evaluate branch
git checkout evaluate
git add scripts/evaluate_private.py
git commit -m "Add evaluation script"

# On utils_wandb branch
git checkout utils_wandb
git add scripts/wandb_setup.py
git commit -m "Add W&B utilities"
```

## Recommended Workflow

1. **Start on main**: All shared scripts and configs are here
2. **Create feature branches**: Each branch adds its specific training/evaluation script
3. **Merge back to main**: Once tested, merge branches back to main for a complete pipeline

