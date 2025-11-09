"""
Evaluate YOLOv12 Model
======================

Evaluates a trained YOLOv12 model on test splits for both private and non-private datasets.
Tracks: mAP(small), mAP(medium), mAP(large), mAP, mAP50, mAP75, mAP50:95

Usage:
    python scripts/evaluate_private.py \
        --checkpoint /content/drive/MyDrive/yolov12_runs/private/checkpoints/best.pt \
        --nonprivate_yaml data/nonprivate.yaml \
        --private_yaml data/private.yaml \
        --output_dir /content/drive/MyDrive/yolov12_runs/eval
"""

import argparse
import json
import csv
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO


def extract_metrics(val_results):
    """
    Extract all required metrics from validation results.
    
    Args:
        val_results: YOLO validation results object
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    if hasattr(val_results, 'box'):
        box = val_results.box
        
        # Basic mAP metrics
        if hasattr(box, 'map'):
            metrics['map'] = float(box.map)
        if hasattr(box, 'map50'):
            metrics['map50'] = float(box.map50)
        if hasattr(box, 'map75'):
            metrics['map75'] = float(box.map75)
        if hasattr(box, 'map50_95'):
            metrics['map50_95'] = float(box.map50_95)
        
        # Size-specific mAPs
        if hasattr(box, 'maps'):
            maps = box.maps
            if len(maps) >= 3:
                metrics['map_small'] = float(maps[0])
                metrics['map_medium'] = float(maps[1])
                metrics['map_large'] = float(maps[2])
            elif len(maps) >= 1:
                # Fallback if size-specific maps not available
                metrics['map_small'] = float(maps[0]) if len(maps) > 0 else 0.0
                metrics['map_medium'] = float(maps[1]) if len(maps) > 1 else 0.0
                metrics['map_large'] = float(maps[2]) if len(maps) > 2 else 0.0
    
    return metrics


def evaluate_model(model, data_yaml, dataset_name, conf=0.001, iou=0.6):
    """
    Evaluate model on a dataset.
    
    Args:
        model: YOLO model instance
        data_yaml: Path to dataset YAML file
        dataset_name: Name of dataset (for logging)
        conf: Confidence threshold
        iou: IoU threshold for NMS
    
    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*60}")
    print(f"Evaluating on {dataset_name} dataset")
    print(f"{'='*60}")
    print(f"Data YAML: {data_yaml}")
    
    # Run validation
    val_results = model.val(
        data=str(data_yaml),
        conf=conf,
        iou=iou,
        verbose=True
    )
    
    # Extract metrics
    metrics = extract_metrics(val_results)
    
    print(f"\nResults for {dataset_name}:")
    print(f"  mAP: {metrics.get('map', 'N/A'):.4f}")
    print(f"  mAP50: {metrics.get('map50', 'N/A'):.4f}")
    print(f"  mAP75: {metrics.get('map75', 'N/A'):.4f}")
    print(f"  mAP50:95: {metrics.get('map50_95', 'N/A'):.4f}")
    print(f"  mAP (small): {metrics.get('map_small', 'N/A'):.4f}")
    print(f"  mAP (medium): {metrics.get('map_medium', 'N/A'):.4f}")
    print(f"  mAP (large): {metrics.get('map_large', 'N/A'):.4f}")
    
    return metrics


def save_results(results, output_dir, checkpoint_path):
    """
    Save evaluation results to JSON and CSV.
    
    Args:
        results: Dictionary of results (dataset_name -> metrics)
        output_dir: Directory to save results
        checkpoint_path: Path to checkpoint used for evaluation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save JSON
    json_path = output_dir / f"evaluation_results_{timestamp}.json"
    json_data = {
        'checkpoint': str(checkpoint_path),
        'timestamp': timestamp,
        'results': results
    }
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"\nResults saved to: {json_path}")
    
    # Save CSV
    csv_path = output_dir / f"evaluation_results_{timestamp}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        header = ['dataset', 'map', 'map50', 'map75', 'map50_95', 'map_small', 'map_medium', 'map_large']
        writer.writerow(header)
        
        # Data rows
        for dataset_name, metrics in results.items():
            row = [
                dataset_name,
                metrics.get('map', ''),
                metrics.get('map50', ''),
                metrics.get('map75', ''),
                metrics.get('map50_95', ''),
                metrics.get('map_small', ''),
                metrics.get('map_medium', ''),
                metrics.get('map_large', ''),
            ]
            writer.writerow(row)
    
    print(f"CSV saved to: {csv_path}")
    
    return json_path, csv_path


def main():
    parser = argparse.ArgumentParser(description='Evaluate YOLOv12 model on test sets')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--nonprivate_yaml', type=str, required=True,
                        help='Path to non-private dataset YAML')
    parser.add_argument('--private_yaml', type=str, required=True,
                        help='Path to private dataset YAML')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save evaluation results')
    parser.add_argument('--conf', type=float, default=0.001,
                        help='Confidence threshold (default: 0.001)')
    parser.add_argument('--iou', type=float, default=0.6,
                        help='IoU threshold for NMS (default: 0.6)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model from: {args.checkpoint}")
    if not Path(args.checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    model = YOLO(args.checkpoint)
    
    # Evaluate on both datasets
    results = {}
    
    # Evaluate on non-private test set
    try:
        nonprivate_metrics = evaluate_model(
            model,
            args.nonprivate_yaml,
            'nonprivate',
            conf=args.conf,
            iou=args.iou
        )
        results['nonprivate'] = nonprivate_metrics
    except Exception as e:
        print(f"Error evaluating on non-private dataset: {e}")
        results['nonprivate'] = {'error': str(e)}
    
    # Evaluate on private test set
    try:
        private_metrics = evaluate_model(
            model,
            args.private_yaml,
            'private',
            conf=args.conf,
            iou=args.iou
        )
        results['private'] = private_metrics
    except Exception as e:
        print(f"Error evaluating on private dataset: {e}")
        results['private'] = {'error': str(e)}
    
    # Save results
    save_results(results, args.output_dir, args.checkpoint)
    
    # Print summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"\nNon-private dataset:")
    if 'error' not in results['nonprivate']:
        for key, value in results['nonprivate'].items():
            print(f"  {key}: {value:.4f}")
    else:
        print(f"  Error: {results['nonprivate']['error']}")
    
    print(f"\nPrivate dataset:")
    if 'error' not in results['private']:
        for key, value in results['private'].items():
            print(f"  {key}: {value:.4f}")
    else:
        print(f"  Error: {results['private']['error']}")
    
    print("\nEvaluation completed!")


if __name__ == '__main__':
    main()

