#!/bin/bash

# ============================================
# Experiment 2: Modern YOLOv12 Optimized Protocol
# ============================================
# - 1 seed (42)
# - 100 epochs
# - Adam optimizer
# - Heavy augmentation (mosaic, mixup, etc.)
# ============================================

echo "============================================"
echo "Experiment 2: Modern YOLOv12 Protocol (Optimized)"
echo "============================================"
echo ""
echo "Settings:"
echo "  - 1 seed (42)"
echo "  - 100 epochs"
echo "  - Adam optimizer"
echo "  - Heavy augmentation"
echo "  - K-shot: 1, 3, 5, 10"
echo ""

cd /content/yolov12/vizwiz_pipeline

seed=42

# First, create K-shot splits
python scripts/create_fewshot_splits.py \
    --input_json /content/datasets/private/merged_private.json \
    --output_dir /content/drive/MyDrive/yolov12_runs/modern_protocol/fewshot_splits \
    --base_images_dir /content/datasets/private \
    --k_shots 1 3 5 10 \
    --seed ${seed}

# Then train each K-shot model
for k in 1 3 5 10; do
    echo ""
    echo "--- Training ${k}-shot model (modern protocol) ---"
    
    python scripts/train_fewshot.py \
        --k_shot ${k} \
        --checkpoint /content/drive/MyDrive/yolov12_runs/nonprivate/nonprivate_training/weights/best.pt \
        --data_yaml /content/drive/MyDrive/yolov12_runs/modern_protocol/fewshot_splits/private_${k}shot.yaml \
        --project_dir /content/drive/MyDrive/yolov12_runs/modern_protocol \
        --epochs 100 \
        --batch_size 4 \
        --optimizer Adam \
        --protocol modern \
        --seed ${seed}
done

echo ""
echo "============================================"
echo "✅ Modern protocol completed!"
echo "============================================"

