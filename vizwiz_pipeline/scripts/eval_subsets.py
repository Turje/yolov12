"""
Evaluate Model on Multiple Subsets
===================================

Evaluates a trained model on multiple subsets (e.g., originals, augmented, combined)
and logs results to Weights & Biases with per-class AP metrics.

Usage:
    python scripts/eval_subsets.py \
        --weights /content/yolov12_runs/private/checkpoints/best.pt \
        --subsets /content/splits/private/val_originals.txt \
                  /content/splits/private/val_augmented.txt \
                  /content/splits/private/test.txt \
        --data_yaml /content/vizwiz/data/private.yaml \
        --wandb_project vizwiz_yolov12 \
        --run_name private_eval
"""

import argparse
import sys
from pathlib import Path
import wandb
import yaml

# Import YOLO
try:
    from scripts.import_yolo import YOLO
except ImportError:
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from yolov12 import YOLO
        print("✅ Using custom YOLOv12 implementation")
    except ImportError:
        try:
            from ultralytics import YOLO
            print("⚠️  Warning: Using ultralytics instead of custom YOLOv12")
        except ImportError:
            raise ImportError("Could not import YOLO")

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.wandb_setup import init_wandb


def evaluate_subset(model, subset_path, subset_name, class_names, img_size=800):
    """
    Evaluate model on a specific subset.
    
    Args:
        model: YOLO model instance
        subset_path: Path to subset list file or YAML
        subset_name: Name for logging (e.g., 'val_originals')
        class_names: Dictionary mapping class IDs to names
        img_size: Image size for evaluation
    
    Returns:
        Dictionary of metrics
    """
    print(f"\n=== Evaluating on {subset_name} ===")
    
    try:
        # Run validation
        results = model.val(data=str(subset_path), imgsz=img_size)
        
        metrics = {}
        
        if hasattr(results, 'box'):
            metrics = {
                f'{subset_name}/map': float(results.box.map),
                f'{subset_name}/map50': float(results.box.map50),
                f'{subset_name}/map75': float(results.box.map75),
            }
            
            print(f"  mAP: {results.box.map:.4f}")
            print(f"  mAP50: {results.box.map50:.4f}")
            print(f"  mAP75: {results.box.map75:.4f}")
            
            # Extract per-class AP
            if hasattr(results.box, 'maps'):
                per_class_ap = results.box.maps
                if len(per_class_ap) > 0:
                    class_ap_data = []
                    for class_id, ap in enumerate(per_class_ap):
                        class_name = class_names.get(class_id, f"class_{class_id}")
                        class_ap_data.append([class_name, float(ap)])
                        metrics[f'{subset_name}/per_class_ap/{class_name}'] = float(ap)
                    
                    # Sort and show top/bottom classes
                    class_ap_data_sorted = sorted(class_ap_data, key=lambda x: x[1], reverse=True)
                    
                    print(f"\n  Top 3 Classes:")
                    for i, (name, ap) in enumerate(class_ap_data_sorted[:3]):
                        print(f"    {i+1}. {name}: {ap:.4f}")
                    
                    if len(class_ap_data_sorted) > 3:
                        print(f"  Bottom 3 Classes:")
                        for i, (name, ap) in enumerate(class_ap_data_sorted[-3:]):
                            print(f"    {i+1}. {name}: {ap:.4f}")
                    
                    # Return table data for W&B
                    metrics['_class_ap_table'] = class_ap_data_sorted
        
        return metrics
        
    except Exception as e:
        print(f"Error evaluating {subset_name}: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on multiple subsets')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to model weights')
    parser.add_argument('--subsets', type=str, nargs='+', required=True,
                        help='List of subset files (txt or yaml) to evaluate')
    parser.add_argument('--data_yaml', type=str, required=True,
                        help='Path to main dataset YAML (for class names)')
    parser.add_argument('--wandb_project', type=str, default='vizwiz_yolov12',
                        help='W&B project name')
    parser.add_argument('--run_name', type=str, default='subset_eval',
                        help='W&B run name')
    parser.add_argument('--img_size', type=int, default=800,
                        help='Image size for evaluation (default: 800)')
    
    args = parser.parse_args()
    
    # Check weights exist
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")
    
    # Load class names from data YAML
    print(f"Loading class names from: {args.data_yaml}")
    with open(args.data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    class_names = data_config.get('names', {})
    
    # Initialize W&B
    print("Initializing Weights & Biases...")
    init_wandb(project_name=args.wandb_project)
    
    run = wandb.init(
        project=args.wandb_project,
        name=args.run_name,
        tags=['evaluation', 'subset-eval'],
        config={
            'weights': str(args.weights),
            'subsets': [str(s) for s in args.subsets],
            'img_size': args.img_size,
            'num_classes': len(class_names),
        }
    )
    
    # Load model
    print(f"Loading model from: {args.weights}")
    model = YOLO(args.weights)
    
    # Evaluate on each subset
    all_metrics = {}
    all_tables = {}
    
    for subset_path in args.subsets:
        subset_path = Path(subset_path)
        if not subset_path.exists():
            print(f"Warning: Subset file not found: {subset_path}")
            continue
        
        subset_name = subset_path.stem
        metrics = evaluate_subset(model, subset_path, subset_name, class_names, args.img_size)
        
        # Extract table data if present
        if '_class_ap_table' in metrics:
            all_tables[subset_name] = metrics.pop('_class_ap_table')
        
        all_metrics.update(metrics)
    
    # Log all metrics to W&B
    if all_metrics:
        wandb.log(all_metrics)
        for key, value in all_metrics.items():
            if isinstance(value, (int, float)):
                run.summary[key] = value
    
    # Log per-class AP tables
    for subset_name, table_data in all_tables.items():
        table = wandb.Table(columns=["Class", "AP"], data=table_data)
        wandb.log({f"{subset_name}_per_class_ap_table": table})
    
    print("\n=== Evaluation Summary ===")
    for subset_path in args.subsets:
        subset_name = Path(subset_path).stem
        if f'{subset_name}/map' in all_metrics:
            print(f"{subset_name}:")
            print(f"  mAP: {all_metrics[f'{subset_name}/map']:.4f}")
            print(f"  mAP50: {all_metrics[f'{subset_name}/map50']:.4f}")
    
    print(f"\nW&B run: {run.url}")
    wandb.finish()
    
    print("Evaluation completed!")


if __name__ == '__main__':
    main()

