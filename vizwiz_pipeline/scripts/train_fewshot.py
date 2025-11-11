#!/usr/bin/env python3
"""
Train YOLOv12 on K-shot private dataset for few-shot learning.

This script fine-tunes from the non-private baseline on K images per class.

Usage:
    python scripts/train_fewshot.py \
        --k_shot 1 \
        --checkpoint /content/drive/MyDrive/yolov12_runs/nonprivate/checkpoints/best.pt \
        --data_yaml /content/datasets/private_fewshot/private_1shot.yaml \
        --project_dir /content/drive/MyDrive/yolov12_runs/fewshot
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure custom YOLOv12 is imported (not ultralytics pip package)
if Path('/content/yolov12').exists():
    sys.path.insert(0, '/content/yolov12')

from ultralytics import YOLO, settings
import wandb


def train_fewshot(
    k_shot,
    checkpoint,
    data_yaml,
    project_dir,
    wandb_project="vizwiz_yolov12",
    model_size="m",
    img_size=800,
    epochs=100,
    patience=50,
    batch_size=4,
    freeze_layers=10,
    lr0=0.001,
    optimizer='Adam',
    protocol='modern',
    seed=42
):
    """
    Train K-shot model with two protocol options.
    
    Protocol Options:
    - 'paper': Match BIV-Priv-Seg exactly (20 epochs, SGD, minimal aug)
    - 'modern': Optimized YOLOv12 (100 epochs, Adam, heavy aug)
    
    Args:
        k_shot: Number of shots per class (1, 3, 5, 10)
        checkpoint: Path to non-private baseline checkpoint
        data_yaml: Path to K-shot YAML config
        project_dir: Output directory for checkpoints
        wandb_project: W&B project name
        model_size: YOLO model size (n/s/m/l/x)
        img_size: Input image size
        epochs: Training epochs (20 for paper, 100 for modern)
        patience: Early stopping patience
        batch_size: Batch size (keep small for few-shot)
        freeze_layers: Number of layers to freeze (10 = backbone stem only)
        lr0: Initial learning rate (0.001)
        optimizer: 'SGD' (paper) or 'Adam' (modern)
        protocol: 'paper' or 'modern'
        seed: Random seed
    """
    
    print(f"\n{'='*60}")
    print(f"Training {k_shot}-shot Model (Protocol: {protocol.upper()})")
    print(f"{'='*60}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Data YAML: {data_yaml}")
    print(f"Output directory: {project_dir}")
    print(f"W&B project: {wandb_project}")
    print(f"K-shot: {k_shot}")
    print(f"Epochs: {epochs}")
    print(f"Optimizer: {optimizer}")
    print(f"Freeze layers: {freeze_layers}")
    print(f"Learning rate: {lr0}")
    
    # Initialize W&B
    os.environ['WANDB_MODE'] = 'online'
    
    wandb.init(
        project=wandb_project,
        name=f"{k_shot}shot_yolov12{model_size}_{protocol}_seed{seed}",
        config={
            'k_shot': k_shot,
            'architecture': f'yolov12{model_size}',
            'checkpoint': str(checkpoint),
            'img_size': img_size,
            'epochs': epochs,
            'batch_size': batch_size,
            'lr0': lr0,
            'freeze_layers': freeze_layers,
            'optimizer': optimizer,
            'protocol': protocol,
            'seed': seed,
            'training_type': 'few_shot'
        },
        settings=wandb.Settings(start_method="thread")
    )
    
    print(f"\nW&B run initialized: {wandb.run.name}")
    print(f"W&B URL: {wandb.run.url}")
    print(f"W&B syncing: {wandb.run.settings.mode}")
    
    # Load model from checkpoint
    print(f"\nLoading checkpoint from: {checkpoint}")
    model = YOLO(checkpoint)
    print(f"✅ Model loaded: YOLOv12-{model_size}")
    
    # Force Ultralytics to use our W&B run
    settings.update({'wandb': True})
    
    # Train with few-shot optimized settings
    print(f"\n{'='*60}")
    print(f"Starting {k_shot}-shot training...")
    print(f"{'='*60}")
    
    # Configure augmentation based on protocol
    if protocol == 'paper':
        # Minimal augmentation (match BIV-Priv-Seg)
        aug_config = {
            'hsv_h': 0.015,      # Minimal hue
            'hsv_s': 0.01,       # Minimal saturation
            'hsv_v': 0.4,        # Brightness only
            'degrees': 0,        # No rotation
            'translate': 0,      # No translation
            'scale': 0,          # No scale
            'shear': 0,          # No shear
            'perspective': 0,    # No perspective
            'flipud': 0.0,       # No vertical flip
            'fliplr': 0.5,       # Horizontal flip only
            'bgr': 0.0,
            'mosaic': 0.0,       # No mosaic
            'mixup': 0.0,        # No mixup
            'copy_paste': 0.0,   # No copy-paste
            'auto_augment': 'none',
            'erasing': 0.0,      # No erasing
        }
    else:  # 'modern'
        # Heavy augmentation (YOLOv12 optimized)
        aug_config = {
            'hsv_h': 0.02,
            'hsv_s': 0.8,
            'hsv_v': 0.5,
            'degrees': 15,
            'translate': 0.15,
            'scale': 0.6,
            'shear': 3.0,
            'perspective': 0.0002,
            'flipud': 0.0,
            'fliplr': 0.5,
            'bgr': 0.0,
            'mosaic': 1.0,
            'mixup': 0.2,
            'copy_paste': 0.2,
            'copy_paste_mode': 'flip',
            'auto_augment': 'randaugment',
            'erasing': 0.5,
        }
    
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=patience,
        batch=batch_size,
        imgsz=img_size,
        project=str(project_dir),
        name=f"{k_shot}shot_training_{protocol}_seed{seed}",
        exist_ok=True,
        pretrained=False,
        optimizer=optimizer,
        verbose=True,
        seed=seed,
        deterministic=True,
        amp=True,
        freeze=freeze_layers,
        lr0=lr0,
        lrf=0.01,
        momentum=0.937 if optimizer == 'SGD' else 0.9,
        weight_decay=0.0001 if protocol == 'paper' else 0.0005,
        warmup_epochs=3.0,
        warmup_momentum=0.8 if optimizer == 'SGD' else 0.85,
        warmup_bias_lr=0.1,
        box=7.5,
        cls=0.5,
        dfl=1.5,
        save_json=True,
        crop_fraction=1.0,
        **aug_config
    )
    
    # Validation on best model
    print(f"\n{'='*60}")
    print(f"Evaluating best {k_shot}-shot model...")
    print(f"{'='*60}")
    
    best_model_path = Path(project_dir) / f"{k_shot}shot_training" / "weights" / "best.pt"
    if best_model_path.exists():
        best_model = YOLO(str(best_model_path))
        metrics = best_model.val(data=str(data_yaml))
        
        # Extract key metrics
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        map_per_class = metrics.box.maps  # Per-class mAP50-95
        
        print(f"\n{'='*60}")
        print(f"✅ {k_shot}-shot Training Complete!")
        print(f"{'='*60}")
        print(f"mAP50: {map50:.4f}")
        print(f"mAP50-95: {map50_95:.4f}")
        
        # Log to W&B
        wandb.log({
            f'{k_shot}shot/final_map50': map50,
            f'{k_shot}shot/final_map50_95': map50_95,
        })
        
        # Per-class breakdown
        if map_per_class is not None and len(map_per_class) > 0:
            print(f"\nPer-class AP (mAP50-95):")
            class_names = model.names
            class_ap_pairs = [(class_names[i], map_per_class[i]) for i in range(len(map_per_class))]
            class_ap_pairs_sorted = sorted(class_ap_pairs, key=lambda x: x[1], reverse=True)
            
            # Top 5
            print(f"\n  Top 5 classes:")
            for i, (name, ap) in enumerate(class_ap_pairs_sorted[:5]):
                print(f"    {i+1}. {name:30s}: {ap:.4f}")
            
            # Bottom 5
            print(f"\n  Bottom 5 classes:")
            for i, (name, ap) in enumerate(class_ap_pairs_sorted[-5:]):
                print(f"    {i+1}. {name:30s}: {ap:.4f}")
            
            # Log per-class AP to W&B table
            wandb.log({
                f'{k_shot}shot/per_class_ap': wandb.Table(
                    columns=['Class', 'AP50-95'],
                    data=[[name, ap] for name, ap in class_ap_pairs_sorted]
                )
            })
    
    wandb.finish()
    
    print(f"\n{'='*60}")
    print(f"✅ {k_shot}-shot training completed!")
    print(f"Best model: {best_model_path}")
    print(f"{'='*60}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Train K-shot YOLOv12 model')
    parser.add_argument('--k_shot', type=int, required=True,
                        help='Number of shots per class (1, 3, 5, 10)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to non-private baseline checkpoint')
    parser.add_argument('--data_yaml', type=str, required=True,
                        help='Path to K-shot YAML config')
    parser.add_argument('--project_dir', type=str, 
                        default='/content/drive/MyDrive/yolov12_runs/fewshot',
                        help='Output directory for checkpoints')
    parser.add_argument('--wandb_project', type=str, default='vizwiz_yolov12',
                        help='W&B project name')
    parser.add_argument('--model_size', type=str, default='m',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLO model size')
    parser.add_argument('--img_size', type=int, default=800,
                        help='Input image size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs (default: 100 for few-shot)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size (keep small for few-shot)')
    parser.add_argument('--freeze_layers', type=int, default=10,
                        help='Number of layers to freeze (10 = backbone stem)')
    parser.add_argument('--lr0', type=float, default=0.001,
                        help='Initial learning rate (low for few-shot)')
    parser.add_argument('--optimizer', type=str, default='Adam',
                        choices=['Adam', 'SGD'],
                        help='Optimizer (Adam=modern, SGD=paper)')
    parser.add_argument('--protocol', type=str, default='modern',
                        choices=['paper', 'modern'],
                        help='Training protocol (paper=BIV-Priv-Seg match, modern=optimized)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    args = parser.parse_args()
    
    # Validate inputs
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    data_yaml = Path(args.data_yaml)
    if not data_yaml.exists():
        raise FileNotFoundError(f"Data YAML not found: {data_yaml}")
    
    # Train
    train_fewshot(
        k_shot=args.k_shot,
        checkpoint=checkpoint,
        data_yaml=data_yaml,
        project_dir=args.project_dir,
        wandb_project=args.wandb_project,
        model_size=args.model_size,
        img_size=args.img_size,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        freeze_layers=args.freeze_layers,
        lr0=args.lr0,
        optimizer=args.optimizer,
        protocol=args.protocol,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

