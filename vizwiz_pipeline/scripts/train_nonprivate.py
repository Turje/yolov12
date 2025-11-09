"""
Train YOLOv12 on Non-Private Dataset
=====================================

Trains YOLOv12 on base_images dataset (100 categories).
Uses pretrained weights and logs to Weights & Biases.

Training Configuration:
- Model size: m (Medium)
- Image size: 800
- Max epochs: 250
- Early stopping patience: 50
- Batch size: 8

Usage:
    python scripts/train_nonprivate.py \
        --data_yaml data/nonprivate.yaml \
        --project_dir /content/drive/MyDrive/yolov12_runs/nonprivate \
        --wandb_project vizwiz_yolov12
"""

import argparse
import os
import sys
from pathlib import Path
import wandb
from ultralytics import YOLO
import torch

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.wandb_setup import init_wandb, log_model_artifact


def train_nonprivate(
    data_yaml,
    project_dir,
    wandb_project="vizwiz_yolov12",
    model_size="m",
    img_size=800,
    epochs=250,
    patience=50,
    batch_size=8,
    pretrained_weights=None,
    seed=42
):
    """
    Train YOLOv12 on non-private dataset.
    
    Args:
        data_yaml: Path to dataset YAML file
        project_dir: Directory to save training outputs
        wandb_project: W&B project name
        model_size: Model size (n, s, m, l, x)
        img_size: Image size for training
        epochs: Maximum number of epochs
        patience: Early stopping patience
        batch_size: Batch size
        pretrained_weights: Path to pretrained weights (if None, uses default)
        seed: Random seed
    """
    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Initialize W&B
    print("Initializing Weights & Biases...")
    init_wandb(project_name=wandb_project)
    
    # Initialize model
    if pretrained_weights is None:
        # Try YOLOv12 first, fallback to YOLOv11/YOLOv10
        try:
            model = YOLO(f'yolo12{model_size}.pt')
            print(f"Using YOLOv12-{model_size}")
        except:
            try:
                model = YOLO(f'yolo11{model_size}.pt')
                print(f"Using YOLOv11-{model_size} (YOLOv12 not available)")
            except:
                model = YOLO(f'yolo10{model_size}.pt')
                print(f"Using YOLOv10-{model_size} (YOLOv12 not available)")
    else:
        model = YOLO(pretrained_weights)
        print(f"Loading pretrained weights from: {pretrained_weights}")
    
    # Create project directory
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B run
    run = wandb.init(
        project=wandb_project,
        name=f"nonprivate_yolov12{model_size}_{img_size}",
        config={
            "model": f"yolov12{model_size}",
            "dataset": "nonprivate",
            "epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "img_size": img_size,
            "data_yaml": str(data_yaml),
            "seed": seed,
        }
    )
    
    print(f"Starting training on non-private dataset...")
    print(f"Data YAML: {data_yaml}")
    print(f"Output directory: {project_dir}")
    print(f"W&B project: {wandb_project}")
    
    # Train model
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        patience=patience,  # Early stopping
        batch=batch_size,
        imgsz=img_size,
        project=str(project_dir),
        name="nonprivate_training",
        exist_ok=True,
        pretrained=True,
        optimizer="SGD",
        verbose=True,
        seed=seed,
        deterministic=True,
        amp=True,  # Automatic Mixed Precision
        
        # Augmentation settings optimized for BLV user photos
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        perspective=0.0001,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.1,
    )
    
    # Get best model path
    best_model_path = project_dir / "nonprivate_training" / "weights" / "best.pt"
    
    if best_model_path.exists():
        # Log best model as W&B artifact
        print(f"Logging best model to W&B: {best_model_path}")
        log_model_artifact(run, str(best_model_path), artifact_name="nonprivate_best_model")
        
        # Also copy to Drive for easy access
        drive_checkpoint_dir = project_dir / "checkpoints"
        drive_checkpoint_dir.mkdir(exist_ok=True)
        import shutil
        shutil.copy2(best_model_path, drive_checkpoint_dir / "best.pt")
        print(f"Checkpoint saved to: {drive_checkpoint_dir / 'best.pt'}")
    else:
        print(f"Warning: Best model not found at {best_model_path}")
    
    # Log final metrics
    if hasattr(results, 'results_dict'):
        metrics = results.results_dict
    else:
        # Try to get metrics from validation
        metrics = {}
        try:
            val_results = model.val()
            if hasattr(val_results, 'box'):
                metrics = {
                    'map': val_results.box.map,
                    'map50': val_results.box.map50,
                    'map75': val_results.box.map75,
                    'map50_95': val_results.box.map50_95,
                }
                # Try to get size-specific maps
                if hasattr(val_results.box, 'maps'):
                    maps = val_results.box.maps
                    if len(maps) >= 3:
                        metrics['map_small'] = maps[0]
                        metrics['map_medium'] = maps[1]
                        metrics['map_large'] = maps[2]
        except Exception as e:
            print(f"Could not extract validation metrics: {e}")
    
    # Log metrics to W&B
    if metrics:
        wandb.log(metrics)
        for key, value in metrics.items():
            run.summary[key] = value
    
    print("Training completed!")
    print(f"Best model: {best_model_path}")
    print(f"W&B run: {run.url}")
    
    wandb.finish()
    
    return results, best_model_path


def main():
    parser = argparse.ArgumentParser(description='Train YOLOv12 on non-private dataset')
    parser.add_argument('--data_yaml', type=str, required=True,
                        help='Path to dataset YAML file')
    parser.add_argument('--project_dir', type=str, required=True,
                        help='Directory to save training outputs')
    parser.add_argument('--wandb_project', type=str, default='vizwiz_yolov12',
                        help='W&B project name (default: vizwiz_yolov12)')
    parser.add_argument('--model_size', type=str, default='m',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (default: m)')
    parser.add_argument('--img_size', type=int, default=800,
                        help='Image size for training (default: 800)')
    parser.add_argument('--epochs', type=int, default=250,
                        help='Maximum number of epochs (default: 250)')
    parser.add_argument('--patience', type=int, default=50,
                        help='Early stopping patience (default: 50)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--pretrained_weights', type=str, default=None,
                        help='Path to pretrained weights (default: uses YOLOv12 pretrained)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    train_nonprivate(
        data_yaml=args.data_yaml,
        project_dir=args.project_dir,
        wandb_project=args.wandb_project,
        model_size=args.model_size,
        img_size=args.img_size,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        pretrained_weights=args.pretrained_weights,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

