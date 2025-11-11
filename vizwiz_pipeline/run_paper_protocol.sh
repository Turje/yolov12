#!/bin/bash

# ============================================
# Experiment 1: Match BIV-Priv-Seg Protocol Exactly
# ============================================
# - 5 random seeds
# - 20 epochs
# - SGD optimizer
# - Minimal augmentation (flip + brightness only)
# ============================================

echo "============================================"
echo "Experiment 1: BIV-Priv-Seg Protocol (Paper Match)"
echo "============================================"
echo ""
echo "Settings:"
echo "  - 5 random seeds (1,2,3,4,5)"
echo "  - 20 epochs"
echo "  - SGD optimizer"
echo "  - Minimal augmentation"
echo "  - K-shot: 1, 3, 5, 10"
echo ""

cd /content/yolov12/vizwiz_pipeline

for seed in 1 2 3 4 5; do
    echo ""
    echo "=========================================="
    echo "Running with seed ${seed}"
    echo "=========================================="
    
    # First, create K-shot splits for this seed
    python scripts/create_fewshot_splits.py \
        --input_json /content/datasets/private/merged_private.json \
        --output_dir /content/drive/MyDrive/yolov12_runs/paper_protocol/seed${seed}/fewshot_splits \
        --base_images_dir /content/datasets/private \
        --k_shots 1 3 5 10 \
        --seed ${seed}
    
    # Then train each K-shot model
    for k in 1 3 5 10; do
        echo ""
        echo "--- Training ${k}-shot model (seed ${seed}) ---"
        
        python scripts/train_fewshot.py \
            --k_shot ${k} \
            --checkpoint /content/drive/MyDrive/yolov12_runs/nonprivate/nonprivate_training/weights/best.pt \
            --data_yaml /content/drive/MyDrive/yolov12_runs/paper_protocol/seed${seed}/fewshot_splits/private_${k}shot.yaml \
            --project_dir /content/drive/MyDrive/yolov12_runs/paper_protocol/seed${seed} \
            --epochs 20 \
            --batch_size 4 \
            --optimizer SGD \
            --protocol paper \
            --seed ${seed}
    done
done

echo ""
echo "============================================"
echo "✅ All 5 seeds completed!"
echo "============================================"
echo ""
echo "Next step: Average results across seeds"
echo "Run: python scripts/average_seed_results.py --protocol paper"

