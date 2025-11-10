"""
Train YOLOv12 on Private Dataset (Fine-tuning)
===============================================

Fine-tunes YOLOv12 on private dataset (16 categories).
Starts from checkpoint trained on non-private dataset.

Training Configuration:
- Model size: m (Medium)
- Image size: 800
- Max epochs: 250
- Early stopping patience: 50
- Batch size: 8

Usage:
    python scripts/train_private.py \
        --data_yaml data/private.yaml \
        --checkpoint /content/drive/MyDrive/yolov12_runs/nonprivate/checkpoints/best.pt \
        --project_dir /content/drive/MyDrive/yolov12_runs/private \
        --wandb_project vizwiz_yolov12
"""

import argparse
import os
import sys
from pathlib import Path
import wandb
import torch

# Import from custom YOLOv12 implementation (not ultralytics)
# Use helper module to find the correct import
try:
    from scripts.import_yolo import YOLO
except ImportError:
    # Fallback: try direct imports
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))  # Go up to yolov12 root
        from yolov12 import YOLO
        print("✅ Using custom YOLOv12 implementation")
    except ImportError:
        try:
            from ultralytics import YOLO
            print("⚠️  Warning: Using ultralytics instead of custom YOLOv12. Install custom repo for better performance.")
        except ImportError:
            raise ImportError("Could not import YOLO. Make sure the custom YOLOv12 repo is installed.")

# Add scripts directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.wandb_setup import init_wandb, log_model_artifact


def train_private(
    data_yaml,
    checkpoint,
    project_dir,
    wandb_project="vizwiz_yolov12",
    model_size="m",
    img_size=800,
    epochs=100,
    patience=20,
    batch_size=8,
    freeze_epochs=5,
    lr_scale=0.1,
    eval_subsets=None,
    seed=42
):
    """
    Fine-tune YOLOv12 on private dataset with two-phase training.
    
    Args:
        data_yaml: Path to dataset YAML file
        checkpoint: Path to checkpoint from non-private training
        project_dir: Directory to save training outputs
        wandb_project: W&B project name
        model_size: Model size (n, s, m, l, x)
        img_size: Image size for training
        epochs: Maximum number of epochs (total)
        patience: Early stopping patience
        batch_size: Batch size
        freeze_epochs: Number of epochs to freeze backbone (phase 1)
        lr_scale: Learning rate scale for phase 2 (after unfreeze)
        eval_subsets: List of subset YAML/list files for domain-wise eval
        seed: Random seed
    """
    # Set random seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Initialize W&B
    print("Initializing Weights & Biases...")
    # Force online mode
    os.environ['WANDB_MODE'] = 'online'
    init_wandb(project_name=wandb_project)
    
    # Load model from checkpoint
    if not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    
    print(f"Loading checkpoint from: {checkpoint}")
    model = YOLO(checkpoint)
    
    # Create project directory
    project_dir = Path(project_dir)
    project_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize W&B run with explicit settings for chart logging
    run = wandb.init(
        project=wandb_project,
        name=f"private_yolov12{model_size}_{img_size}_finetune",
        tags=["private-finetune", "combined"],
        config={
            "model": f"yolov12{model_size}",
            "dataset": "private",
            "checkpoint": str(checkpoint),
            "epochs": epochs,
            "freeze_epochs": freeze_epochs,
            "lr_scale": lr_scale,
            "patience": patience,
            "batch_size": batch_size,
            "img_size": img_size,
            "data_yaml": str(data_yaml),
            "seed": seed,
        },
        settings=wandb.Settings(start_method="thread")  # Ensure proper integration
    )
    
    print(f"W&B run initialized: {run.name}")
    print(f"W&B URL: {run.url}")
    print(f"W&B syncing: {wandb.run._settings.mode}")  # Should show 'online'
    
    print(f"Starting two-phase fine-tuning on private dataset...")
    print(f"Data YAML: {data_yaml}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Output directory: {project_dir}")
    print(f"W&B project: {wandb_project}")
    print(f"Phase 1: Freeze backbone for {freeze_epochs} epochs")
    print(f"Phase 2: Unfreeze with lr_scale={lr_scale} for {epochs - freeze_epochs} epochs")
    
    # Force Ultralytics to use our W&B run (not create a new one)
    from ultralytics import settings
    settings.update({'wandb': True})  # Explicitly enable W&B
    
    # Phase 1: Train with frozen backbone
    print("\n=== Phase 1: Training with frozen backbone ===")
    results_phase1 = model.train(
        data=str(data_yaml),
        epochs=freeze_epochs,
        patience=epochs,  # No early stopping in phase 1
        batch=batch_size,
        imgsz=img_size,
        project=str(project_dir),
        name="private_training",
        exist_ok=True,
        pretrained=False,  # Already using checkpoint
        optimizer="SGD",
        verbose=True,
        seed=seed,
        deterministic=True,
        amp=True,  # Automatic Mixed Precision
        freeze=10,  # Freeze only first 10 layers (backbone stem + early layers)
        save_json=True,  # Enable COCO evaluation for mAP-S/M/L
        
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
    
    # Phase 2: Continue training with unfrozen backbone and scaled LR
    if epochs > freeze_epochs:
        print(f"\n=== Phase 2: Training with unfrozen backbone (lr × {lr_scale}) ===")
        # Get the last checkpoint from phase 1
        last_checkpoint = project_dir / "private_training" / "weights" / "last.pt"
        if not last_checkpoint.exists():
            print("Warning: Phase 1 checkpoint not found, skipping phase 2")
            results = results_phase1
        else:
            # Load model from phase 1 checkpoint
            model_phase2 = YOLO(str(last_checkpoint))
            
            # Calculate remaining epochs for phase 2
            remaining_epochs = epochs - freeze_epochs
            
            results = model_phase2.train(
                data=str(data_yaml),
                epochs=remaining_epochs,  # Only remaining epochs
                patience=patience,  # Early stopping
                batch=4,  # Reduced batch size for unfrozen training (more memory needed)
                imgsz=img_size,
                project=str(project_dir),
                name="private_training2",  # Different name to avoid conflict
                exist_ok=True,
                pretrained=False,
                optimizer="SGD",
                verbose=True,
                seed=seed,
                deterministic=True,
                amp=True,
                resume=False,  # Start fresh, not resume
                freeze=None,  # Unfreeze all layers
                lr0=0.01 * lr_scale,  # Scale learning rate
                save_json=True,  # Enable COCO evaluation for mAP-S/M/L
                
                # Same augmentation settings
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
    else:
        results = results_phase1
    
    # Get best model path
    best_model_path = project_dir / "private_training" / "weights" / "best.pt"
    
    if best_model_path.exists():
        # Log best model as W&B artifact
        print(f"Logging best model to W&B: {best_model_path}")
        log_model_artifact(run, str(best_model_path), artifact_name="private_best_model")
        
        # Also copy to Drive for easy access
        drive_checkpoint_dir = project_dir / "checkpoints"
        drive_checkpoint_dir.mkdir(exist_ok=True)
        import shutil
        shutil.copy2(best_model_path, drive_checkpoint_dir / "best.pt")
        print(f"Checkpoint saved to: {drive_checkpoint_dir / 'best.pt'}")
    else:
        print(f"Warning: Best model not found at {best_model_path}")
    
    # Evaluate on combined validation set and extract per-class metrics
    print("\n=== Evaluating on combined validation set ===")
    try:
        import yaml
        # Load data YAML to get class names
        with open(data_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
        class_names = data_config.get('names', {})
        
        # Run validation
        val_results = model.val(data=str(data_yaml))
        
        if hasattr(val_results, 'box'):
            metrics = {
                'final/map': val_results.box.map,
                'final/map50': val_results.box.map50,
                'final/map75': val_results.box.map75,
            }
            
            # Extract per-class AP
            if hasattr(val_results.box, 'maps'):
                per_class_ap = val_results.box.maps
                if len(per_class_ap) > 0:
                    # Create per-class AP table
                    class_ap_data = []
                    for class_id, ap in enumerate(per_class_ap):
                        class_name = class_names.get(class_id, f"class_{class_id}")
                        class_ap_data.append([class_name, float(ap)])
                        metrics[f'per_class_ap/{class_name}'] = float(ap)
                    
                    # Sort by AP to find top/bottom classes
                    class_ap_data_sorted = sorted(class_ap_data, key=lambda x: x[1], reverse=True)
                    
                    print("\n=== Top 5 Classes by AP ===")
                    for i, (name, ap) in enumerate(class_ap_data_sorted[:5]):
                        print(f"{i+1}. {name}: {ap:.4f}")
                    
                    print("\n=== Bottom 5 Classes by AP ===")
                    for i, (name, ap) in enumerate(class_ap_data_sorted[-5:]):
                        print(f"{i+1}. {name}: {ap:.4f}")
                    
                    # Log table to W&B
                    table = wandb.Table(columns=["Class", "AP"], data=class_ap_data_sorted)
                    wandb.log({"per_class_ap_table": table})
            
            # Log metrics to W&B
            wandb.log(metrics)
            for key, value in metrics.items():
                run.summary[key] = value
        else:
            print("Warning: Could not extract validation metrics from val_results")
    except Exception as e:
        print(f"Could not extract validation metrics: {e}")
        import traceback
        traceback.print_exc()
    
    # Domain-wise evaluation if subsets provided
    if eval_subsets:
        print("\n=== Domain-wise Evaluation ===")
        for subset_path in eval_subsets:
            subset_path = Path(subset_path)
            if not subset_path.exists():
                print(f"Warning: Subset file not found: {subset_path}")
                continue
            
            subset_name = subset_path.stem  # e.g., 'val_originals', 'test_augmented'
            print(f"\nEvaluating on subset: {subset_name}")
            
            try:
                # Run validation on this subset
                subset_results = model.val(data=str(subset_path))
                
                if hasattr(subset_results, 'box'):
                    subset_metrics = {
                        f'{subset_name}/map': subset_results.box.map,
                        f'{subset_name}/map50': subset_results.box.map50,
                        f'{subset_name}/map75': subset_results.box.map75,
                    }
                    
                    wandb.log(subset_metrics)
                    for key, value in subset_metrics.items():
                        run.summary[key] = value
                    
                    print(f"  mAP: {subset_results.box.map:.4f}")
                    print(f"  mAP50: {subset_results.box.map50:.4f}")
            except Exception as e:
                print(f"Error evaluating subset {subset_name}: {e}")
    
    print("Fine-tuning completed!")
    print(f"Best model: {best_model_path}")
    print(f"W&B run: {run.url}")
    
    wandb.finish()
    
    return results, best_model_path


def main():
    parser = argparse.ArgumentParser(description='Fine-tune YOLOv12 on private dataset with two-phase training')
    parser.add_argument('--data_yaml', type=str, required=True,
                        help='Path to dataset YAML file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint from non-private training')
    parser.add_argument('--project_dir', type=str, required=True,
                        help='Directory to save training outputs')
    parser.add_argument('--wandb_project', type=str, default='vizwiz_yolov12',
                        help='W&B project name (default: vizwiz_yolov12)')
    parser.add_argument('--model_size', type=str, default='m',
                        choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (default: m)')
    parser.add_argument('--img_size', type=int, default=800,
                        help='Image size for training (default: 800)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum number of epochs total (default: 100)')
    parser.add_argument('--freeze_epochs', type=int, default=5,
                        help='Number of epochs to freeze backbone (default: 5)')
    parser.add_argument('--lr_scale', type=float, default=0.1,
                        help='Learning rate scale for phase 2 (default: 0.1)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (default: 20)')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size (default: 8)')
    parser.add_argument('--eval_subsets', type=str, nargs='*', default=None,
                        help='List of subset YAML/list files for domain-wise evaluation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    train_private(
        data_yaml=args.data_yaml,
        checkpoint=args.checkpoint,
        project_dir=args.project_dir,
        wandb_project=args.wandb_project,
        model_size=args.model_size,
        img_size=args.img_size,
        epochs=args.epochs,
        freeze_epochs=args.freeze_epochs,
        lr_scale=args.lr_scale,
        patience=args.patience,
        batch_size=args.batch_size,
        eval_subsets=args.eval_subsets,
        seed=args.seed
    )


if __name__ == '__main__':
    main()

