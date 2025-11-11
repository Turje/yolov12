# ✅ Ready to Start: 100-Epoch Training with W&B

## 🎯 What's Changed

### ✅ **1. W&B Project Name**
- **Old:** `vizwiz_yolov12`
- **New:** `few_shot_query` ✨
- All runs will log to: `https://wandb.ai/[your-username]/few_shot_query`

### ✅ **2. Training Duration**
- **Old:** 20 epochs (~4 hours, under-trained)
- **New:** 100 epochs (~15 hours, 5× more training) 🚀
- **Expected improvement:** 5-15 mAP50 points closer to paper

### ✅ **3. Complete Logging & Saving**
- ✅ All metrics logged to W&B in real-time
- ✅ Training curves visible during training
- ✅ Per-class AP tables logged
- ✅ All checkpoints saved to Google Drive
- ✅ Progress tracking with time estimates

---

## 📦 Files Ready

### Updated Files:
1. ✅ `scripts/train_fewshot.py` - Now uses `few_shot_query` W&B project
2. ✅ `run_paper_protocol_100epochs_WANDB.sh` - 100-epoch script with W&B verification
3. ✅ `COLAB_SETUP_WANDB.md` - Step-by-step setup guide
4. ✅ `EPOCH_COMPARISON.md` - Analysis of why 100 epochs

### Files to Upload to Colab:
```
/content/yolov12/vizwiz_pipeline/scripts/train_fewshot.py
/content/yolov12/vizwiz_pipeline/run_paper_protocol_100epochs_WANDB.sh
```

---

## 🚀 Quick Start (3 Steps)

### Step 1: Upload Files to Colab

**Option A: Manual Upload**
1. Open Colab file browser
2. Navigate to `/content/yolov12/vizwiz_pipeline/scripts/`
3. Upload `train_fewshot.py` (replace existing)
4. Navigate to `/content/yolov12/vizwiz_pipeline/`
5. Upload `run_paper_protocol_100epochs_WANDB.sh`

**Option B: Use `scp` or Google Drive**
1. Upload files to Google Drive first
2. Copy them to Colab:
   ```bash
   !cp /content/drive/MyDrive/[your-folder]/train_fewshot.py /content/yolov12/vizwiz_pipeline/scripts/
   !cp /content/drive/MyDrive/[your-folder]/run_paper_protocol_100epochs_WANDB.sh /content/yolov12/vizwiz_pipeline/
   !chmod +x /content/yolov12/vizwiz_pipeline/run_paper_protocol_100epochs_WANDB.sh
   ```

### Step 2: Configure W&B

```python
import os
import wandb

# Set your W&B API key
os.environ['WANDB_API_KEY'] = "YOUR_KEY_HERE"  # ← Get from https://wandb.ai/authorize
wandb.login(key=os.environ['WANDB_API_KEY'])

print(f"✅ W&B configured")
print(f"   Project: few_shot_query")
print(f"   Dashboard: https://wandb.ai/{wandb.Api().viewer.username}/few_shot_query")
```

### Step 3: Start Training

```bash
!bash /content/yolov12/vizwiz_pipeline/run_paper_protocol_100epochs_WANDB.sh
```

---

## ⏰ Timeline

```
Start:           Now
Seed 1 done:     ~3 hours
Seed 2 done:     ~6 hours
Seed 3 done:     ~9 hours
Seed 4 done:     ~12 hours
Seed 5 done:     ~15 hours
────────────────────────────
Total:           ~15 hours

Average results: +5 minutes
────────────────────────────
Complete:        ~15.1 hours from start
```

**Start now → Results tomorrow morning!** ☀️

---

## 📊 What You'll Get

### W&B Dashboard (`few_shot_query` project):
- 20 training runs (5 seeds × 4 K-shots)
- Live training curves (loss, mAP, precision, recall)
- Per-class AP tables
- Hyperparameter comparison
- Model checkpoints

### Google Drive (`/content/drive/MyDrive/yolov12_runs/paper_protocol_100ep/`):
- `seed1/` through `seed5/` directories
- Each with `1shot_training_paper_seed1/`, `3shot_training_paper_seed1/`, etc.
- `weights/best.pt` - Best model checkpoint
- `weights/last.pt` - Last epoch checkpoint
- `results.csv` - All metrics per epoch

### Comparison Table:
```
| K-shot | Your 20ep | Your 100ep (exp) | Paper (DeFRCN) |
|--------|-----------|------------------|----------------|
| 1      | 5.5       | 12-15            | 18.7           |
| 3      | 7.6       | 18-22            | 26.2           |
| 5      | 15.9      | 26-28            | 30.1           |
| 10     | 21.6      | 32-35            | 35.4           |
```

---

## 🔍 Monitor Progress

### Live W&B Dashboard:
```
https://wandb.ai/[your-username]/few_shot_query/runs
```

### Check Checkpoints:
```bash
!ls -lh /content/drive/MyDrive/yolov12_runs/paper_protocol_100ep/seed*/*/weights/best.pt
```

### Check Current Run:
```bash
!tail -n 20 /content/yolov12/vizwiz_pipeline/wandb/latest-run/logs/debug.log
```

---

## ⚠️ Troubleshooting

### W&B Login Error
```python
# Re-run W&B login
import os
import wandb

os.environ['WANDB_API_KEY'] = "YOUR_KEY"
wandb.login(key=os.environ['WANDB_API_KEY'], relogin=True)
```

### Colab Disconnected
- **Don't panic!** All checkpoints are in Google Drive
- Check what completed:
  ```bash
  !find /content/drive/MyDrive/yolov12_runs/paper_protocol_100ep -name "best.pt" | wc -l
  ```
- Resume from last incomplete seed

### GPU Out of Memory
- Script uses `batch_size=4` (safe for L4/T4)
- If OOM: Restart runtime and reduce batch size to 2

---

## 📝 After Training

1. **Generate averaged results:**
   ```bash
   !python scripts/average_seed_results.py \
     --protocol paper \
     --base_dir /content/drive/MyDrive/yolov12_runs/paper_protocol_100ep
   ```

2. **Download comparison table:**
   - Will be saved to `/content/drive/MyDrive/yolov12_runs/results/`
   - `paper_raw_results.csv`
   - `paper_averaged_results.csv`

3. **Thesis integration:**
   - Include W&B dashboard screenshots
   - Report mean ± std for each K-shot
   - Discuss iteration gap (5× vs 50× paper's training)
   - Compare architectural differences (YOLOv12 vs DeFRCN)

---

## ✅ Checklist

Before starting:
- [ ] Google Drive mounted
- [ ] W&B logged in (check with `wandb whoami`)
- [ ] `train_fewshot.py` updated (contains `few_shot_query`)
- [ ] `run_paper_protocol_100epochs_WANDB.sh` uploaded & executable
- [ ] Non-private checkpoint exists
- [ ] Private dataset (`merged_private.json`) exists
- [ ] ~15 GB free on Google Drive
- [ ] Colab Pro (recommended) or stable connection

---

## 🎉 You're Ready!

Everything is configured for:
- ✅ 100 epochs per K-shot
- ✅ Proper W&B logging to `few_shot_query`
- ✅ All checkpoints saved to Google Drive
- ✅ ~15 hour training time
- ✅ Expected competitive results

**Open Colab and start training!** 🚀

