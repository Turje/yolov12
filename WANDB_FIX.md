# W&B Charts Fix for Step 12

## Add This Cell to Your Colab (Before Step 12)

```python
# ============================================
# Force W&B Online Mode & Verify Integration
# ============================================

import os
import wandb

# 1. Force online syncing (not offline mode)
os.environ['WANDB_MODE'] = 'online'

# 2. Re-login to ensure fresh connection
wandb.login(relogin=True)

# 3. Verify configuration
print("=" * 50)
print("W&B Configuration Check:")
print(f"  Mode: {os.environ.get('WANDB_MODE', 'default')}")
print(f"  Logged in: {wandb.api.api_key is not None}")
print(f"  API Key set: {'WANDB_API_KEY' in os.environ}")
print("=" * 50)

# 4. Test connection
try:
    api = wandb.Api()
    user = api.viewer
    print(f"✅ Connected as: {user.username}")
    print(f"✅ Entity: {user.entity if hasattr(user, 'entity') else 'default'}")
except Exception as e:
    print(f"⚠️  Connection test failed: {e}")
    print("   Try running: wandb.login(relogin=True)")

print("\n✅ W&B ready for training with live charts!")
```

## What This Does

1. **Forces online mode** - Ensures W&B syncs in real-time (not buffered)
2. **Re-authenticates** - Refreshes connection to W&B servers
3. **Verifies connection** - Shows if you're properly logged in
4. **Tests API** - Confirms you can reach W&B servers

## Updated train_private.py

The script now includes:
- `os.environ['WANDB_MODE'] = 'online'` - Forces online syncing
- `settings=wandb.Settings(start_method="thread")` - Better Ultralytics integration
- Debug prints showing run URL and sync mode

## Expected Output When Running Step 12

```
Initializing Weights & Biases...
wandb: Currently logged in as: sturjem000 (...)
W&B initialized for project: vizwiz_yolov12
W&B run initialized: private_yolov12m_800_finetune
W&B URL: https://wandb.ai/your-entity/vizwiz_yolov12/runs/xxxxx
W&B syncing: online                    ← Should say 'online', not 'offline'
```

## Charts You Should See (within 1-2 epochs)

### Training Metrics
- `train/box_loss` - Bounding box regression loss
- `train/cls_loss` - Classification loss  
- `train/dfl_loss` - Distribution focal loss
- `lr/pg0`, `lr/pg1`, `lr/pg2` - Learning rates per parameter group

### Validation Metrics
- `metrics/mAP50` - mAP at IoU 0.5
- `metrics/mAP50-95` - mAP averaged over IoU 0.5-0.95
- `metrics/precision`, `metrics/recall`
- `val/box_loss`, `val/cls_loss`, `val/dfl_loss`

### System Metrics
- `system/gpu_memory` - GPU VRAM usage
- `system/gpu_utilization` - GPU compute %
- `system/cpu_utilization` - CPU usage

### Custom Metrics (from our enhancements)
- `per_class_ap/{class_name}` - AP per class (after validation)
- `val_originals/map`, `val_augmented/map` - Domain-wise metrics
- Tables: `per_class_ap_table` with sortable class performance

## Troubleshooting

### If charts still don't appear:

**1. Check W&B page after 2-3 batches:**
- Click the run URL from training output
- Wait 30 seconds, then refresh browser

**2. Verify Ultralytics integration:**
```python
# After training starts, in a new cell:
import wandb
print(wandb.run)  # Should NOT be None
print(wandb.run.name)  # Should show run name
```

**3. Force sync manually:**
```bash
# In a terminal/shell cell
!wandb sync --sync-all
```

**4. Check for errors in training output:**
- Look for warnings like "wandb: ERROR" or "Failed to upload"
- May indicate network/firewall issues

**5. Fallback - Enable TensorBoard:**
Training also logs to TensorBoard (always works):
```python
%load_ext tensorboard
%tensorboard --logdir /content/yolov12_runs/private/private_training
```

## After Training Completes

Even if live charts didn't work, you can manually log results:

```python
import wandb
import pandas as pd

# Read results CSV
results = pd.read_csv('/content/yolov12_runs/private/private_training/results.csv')

# Create a new W&B run to log retrospectively
run = wandb.init(project='vizwiz_yolov12', name='private_results_manual')

for idx, row in results.iterrows():
    wandb.log({
        'epoch': row['epoch'],
        'train/box_loss': row['train/box_loss'],
        'train/cls_loss': row['train/cls_loss'],
        'metrics/mAP50': row['metrics/mAP50'],
        # ... add other columns
    })

wandb.finish()
```

## Summary

✅ Updated `train_private.py` with forced online mode  
✅ Added connection verification prints  
✅ Better Ultralytics integration settings  
✅ Fallback to TensorBoard if needed  

Run the cell above before starting step 12, and you should see live charts!

