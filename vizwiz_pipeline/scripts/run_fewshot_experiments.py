#!/usr/bin/env python3
"""
Run complete few-shot experimental pipeline.

This script orchestrates:
1. Creating K-shot splits (1, 3, 5, 10-shot)
2. Training all K-shot models
3. Collecting results into comparison table

Usage:
    python scripts/run_fewshot_experiments.py \
        --input_json /content/datasets/private/merged_private.json \
        --base_images_dir /content/datasets/private \
        --checkpoint /content/drive/MyDrive/yolov12_runs/nonprivate/checkpoints/best.pt \
        --output_dir /content/drive/MyDrive/yolov12_runs
"""

import argparse
import json
import subprocess
from pathlib import Path
import pandas as pd


def run_command(cmd, description):
    """Run shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"❌ Error running: {description}")
        raise RuntimeError(f"Command failed with exit code {result.returncode}")
    
    print(f"✅ {description} completed!")
    return result


def create_fewshot_splits(input_json, output_dir, base_images_dir, k_shots=[1, 3, 5, 10], seed=42):
    """Create K-shot splits using create_fewshot_splits.py"""
    cmd = [
        'python', 'scripts/create_fewshot_splits.py',
        '--input_json', str(input_json),
        '--output_dir', str(output_dir),
        '--base_images_dir', str(base_images_dir),
        '--k_shots'] + [str(k) for k in k_shots] + [
        '--seed', str(seed)
    ]
    
    run_command(cmd, "Creating K-shot splits")


def train_fewshot_model(k_shot, checkpoint, fewshot_dir, output_dir, epochs=100, batch_size=4):
    """Train a single K-shot model."""
    data_yaml = Path(fewshot_dir) / f'private_{k_shot}shot.yaml'
    
    cmd = [
        'python', 'scripts/train_fewshot.py',
        '--k_shot', str(k_shot),
        '--checkpoint', str(checkpoint),
        '--data_yaml', str(data_yaml),
        '--project_dir', str(output_dir / 'fewshot'),
        '--epochs', str(epochs),
        '--batch_size', str(batch_size)
    ]
    
    run_command(cmd, f"Training {k_shot}-shot model")


def extract_metrics(results_csv):
    """Extract final metrics from results.csv"""
    if not Path(results_csv).exists():
        return None
    
    df = pd.read_csv(results_csv)
    last_row = df.iloc[-1]
    
    return {
        'map50': last_row.get('metrics/mAP50(B)', None),
        'map50_95': last_row.get('metrics/mAP50-95(B)', None),
        'epoch': last_row.get('epoch', None)
    }


def generate_comparison_table(output_dir, k_shots=[1, 3, 5, 10]):
    """Generate comparison table with all results."""
    
    print(f"\n{'='*60}")
    print(f"Generating Comparison Table")
    print(f"{'='*60}")
    
    # Baseline results from papers (DeFRCN)
    defrcn_results = {
        1: {'map50': 18.7, 'map50_95': 12.3},
        3: {'map50': 26.2, 'map50_95': 18.5},
        5: {'map50': 30.1, 'map50_95': 21.4},
        10: {'map50': 35.4, 'map50_95': 25.8},
        'full': {'map50': 48.9, 'map50_95': 34.2}
    }
    
    # Collect YOLOv12 results
    yolo_results = {}
    for k in k_shots:
        results_csv = output_dir / 'fewshot' / f'{k}shot_training' / 'results.csv'
        metrics = extract_metrics(results_csv)
        if metrics:
            yolo_results[k] = metrics
        else:
            yolo_results[k] = {'map50': 'N/A', 'map50_95': 'N/A'}
    
    # Check for full training results
    full_results_csv = output_dir / 'private' / 'private_training' / 'results.csv'
    if Path(full_results_csv).exists():
        full_metrics = extract_metrics(full_results_csv)
        if full_metrics:
            yolo_results['full'] = full_metrics
        else:
            yolo_results['full'] = {'map50': 'N/A', 'map50_95': 'N/A'}
    else:
        yolo_results['full'] = {'map50': 'TBD', 'map50_95': 'TBD'}
    
    # Create comparison table
    rows = []
    
    # Header row
    rows.append("| Method | Training Size | mAP50 | mAP50-95 | Notes |")
    rows.append("|--------|---------------|-------|----------|-------|")
    
    # Few-shot results
    for k in k_shots:
        n_imgs = k * 16  # 16 classes
        
        # DeFRCN baseline
        defrcn = defrcn_results[k]
        rows.append(f"| DeFRCN (paper) | {k}-shot ({n_imgs} imgs) | {defrcn['map50']:.1f} | {defrcn['map50_95']:.1f} | Faster R-CNN |")
        
        # YOLOv12
        yolo = yolo_results[k]
        map50_str = f"{yolo['map50']:.1f}" if isinstance(yolo['map50'], (int, float)) else yolo['map50']
        map50_95_str = f"{yolo['map50_95']:.1f}" if isinstance(yolo['map50_95'], (int, float)) else yolo['map50_95']
        rows.append(f"| YOLOv12-m (ours) | {k}-shot ({n_imgs} imgs) | {map50_str} | {map50_95_str} | One-stage |")
    
    # Full training results
    defrcn_full = defrcn_results['full']
    rows.append(f"| DeFRCN (paper) | Full (~3,500 imgs) | {defrcn_full['map50']:.1f} | {defrcn_full['map50_95']:.1f} | Faster R-CNN |")
    
    yolo_full = yolo_results['full']
    map50_str = f"{yolo_full['map50']:.1f}" if isinstance(yolo_full['map50'], (int, float)) else yolo_full['map50']
    map50_95_str = f"{yolo_full['map50_95']:.1f}" if isinstance(yolo_full['map50_95'], (int, float)) else yolo_full['map50_95']
    rows.append(f"| YOLOv12-m (ours) | Full (3,451 imgs) | {map50_str} | {map50_95_str} | One-stage |")
    
    table = "\n".join(rows)
    
    # Save to file
    table_file = output_dir / 'comparison_table.md'
    with open(table_file, 'w') as f:
        f.write("# YOLOv12 vs DeFRCN Few-Shot Comparison\n\n")
        f.write("## Results Summary\n\n")
        f.write(table)
        f.write("\n\n## Notes\n\n")
        f.write("- **DeFRCN results**: From VizWiz-FewShot (2022) and BIV-Priv-Seg (2024) papers\n")
        f.write("- **YOLOv12 results**: Our experiments using same protocol and data splits\n")
        f.write("- **mAP50**: Mean Average Precision at IoU=0.50 (higher is better)\n")
        f.write("- **mAP50-95**: Mean Average Precision averaged over IoU=0.50:0.05:0.95 (stricter metric)\n")
    
    print(f"\n{table}")
    print(f"\n✅ Comparison table saved to: {table_file}")
    
    # Also create CSV for easy plotting
    csv_data = []
    for k in k_shots + ['full']:
        training_size = f"{k}-shot" if k != 'full' else 'Full'
        csv_data.append({
            'k_shot': k,
            'training_size': training_size,
            'defrcn_map50': defrcn_results[k]['map50'],
            'defrcn_map50_95': defrcn_results[k]['map50_95'],
            'yolo_map50': yolo_results[k]['map50'],
            'yolo_map50_95': yolo_results[k]['map50_95']
        })
    
    df = pd.DataFrame(csv_data)
    csv_file = output_dir / 'comparison_results.csv'
    df.to_csv(csv_file, index=False)
    print(f"✅ CSV results saved to: {csv_file}")
    
    return table


def main():
    parser = argparse.ArgumentParser(description='Run complete few-shot experimental pipeline')
    parser.add_argument('--input_json', type=str, required=True,
                        help='Path to merged_private.json')
    parser.add_argument('--base_images_dir', type=str, required=True,
                        help='Root directory containing images')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to non-private baseline checkpoint')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for all results')
    parser.add_argument('--k_shots', type=int, nargs='+', default=[1, 3, 5, 10],
                        help='K values to evaluate (default: 1 3 5 10)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs per K-shot model')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--skip_splits', action='store_true',
                        help='Skip creating splits (use existing)')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training (only generate comparison table)')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    fewshot_dir = output_dir / 'fewshot_splits'
    
    print(f"\n{'='*60}")
    print(f"Few-Shot Experimental Pipeline")
    print(f"{'='*60}")
    print(f"Input JSON: {args.input_json}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Output directory: {output_dir}")
    print(f"K-shot values: {args.k_shots}")
    print(f"Epochs per model: {args.epochs}")
    
    # Step 1: Create K-shot splits
    if not args.skip_splits:
        create_fewshot_splits(
            args.input_json,
            fewshot_dir,
            args.base_images_dir,
            args.k_shots,
            args.seed
        )
    else:
        print("\n⏭️  Skipping split creation (using existing)")
    
    # Step 2: Train K-shot models
    if not args.skip_training:
        for k in args.k_shots:
            train_fewshot_model(
                k,
                args.checkpoint,
                fewshot_dir,
                output_dir,
                args.epochs,
                args.batch_size
            )
    else:
        print("\n⏭️  Skipping training (using existing results)")
    
    # Step 3: Generate comparison table
    generate_comparison_table(output_dir, args.k_shots)
    
    print(f"\n{'='*60}")
    print(f"✅ Few-Shot Experimental Pipeline Complete!")
    print(f"{'='*60}")
    print(f"\nResults saved to: {output_dir}")
    print(f"- Comparison table: {output_dir / 'comparison_table.md'}")
    print(f"- CSV results: {output_dir / 'comparison_results.csv'}")
    print(f"- Model checkpoints: {output_dir / 'fewshot'}")


if __name__ == '__main__':
    main()

