#!/usr/bin/env python3
"""
Average few-shot results across multiple random seeds.

Usage:
    python scripts/average_seed_results.py \
        --protocol paper \
        --base_dir /content/drive/MyDrive/yolov12_runs/paper_protocol \
        --seeds 1 2 3 4 5
"""

import argparse
import pandas as pd
from pathlib import Path
import numpy as np


def collect_results(base_dir, seeds, k_shots, protocol='paper'):
    """Collect results from all seed runs."""
    
    results = []
    
    for seed in seeds:
        for k in k_shots:
            # Try multiple possible locations (based on how run_paper_protocol.sh saves)
            possible_paths = [
                # Path 1: Direct under seed directory (how run_paper_protocol.sh actually saves)
                Path(base_dir) / f'seed{seed}' / f'{k}shot_training_{protocol}_seed{seed}' / 'results.csv',
                # Path 2: Under fewshot subdirectory (old structure)
                Path(base_dir) / f'seed{seed}' / 'fewshot' / f'{k}shot_training_{protocol}_seed{seed}' / 'results.csv',
                # Path 3: Without protocol suffix
                Path(base_dir) / f'seed{seed}' / f'{k}shot_training' / 'results.csv',
            ]
            
            results_csv = None
            for path in possible_paths:
                if path.exists():
                    results_csv = path
                    break
            
            if results_csv is None:
                print(f"⚠️  Results not found for seed={seed}, k={k}")
                print(f"    Tried: {possible_paths[0]}")
                continue
            
            # Read last row (best model)
            df = pd.read_csv(results_csv)
            last_row = df.iloc[-1]
            
            results.append({
                'seed': seed,
                'k_shot': k,
                'map50': last_row.get('metrics/mAP50(B)', None),
                'map50_95': last_row.get('metrics/mAP50-95(B)', None),
                'precision': last_row.get('metrics/precision(B)', None),
                'recall': last_row.get('metrics/recall(B)', None)
            })
    
    return pd.DataFrame(results)


def average_results(df_results):
    """Compute mean and std across seeds."""
    
    # Group by k_shot
    grouped = df_results.groupby('k_shot').agg({
        'map50': ['mean', 'std', 'count'],
        'map50_95': ['mean', 'std', 'count'],
        'precision': ['mean', 'std'],
        'recall': ['mean', 'std']
    })
    
    return grouped


def print_comparison_table(averaged, protocol='paper'):
    """Print formatted comparison table."""
    
    # DeFRCN baselines from paper
    defrcn_results = {
        1: {'map50': 18.7, 'map50_95': 12.3},
        3: {'map50': 26.2, 'map50_95': 18.5},
        5: {'map50': 30.1, 'map50_95': 21.4},
        10: {'map50': 35.4, 'map50_95': 25.8}
    }
    
    print("\n" + "="*80)
    print(f"Few-Shot Results Comparison ({protocol.upper()} protocol)")
    print("="*80)
    print("")
    print("| Method | K-shot | mAP50 (mean ± std) | mAP50-95 (mean ± std) | N |")
    print("|--------|--------|--------------------|-----------------------|---|")
    
    for k in sorted(averaged.index):
        # DeFRCN baseline
        defrcn = defrcn_results.get(k, {'map50': 'N/A', 'map50_95': 'N/A'})
        if isinstance(defrcn['map50'], (int, float)):
            print(f"| DeFRCN (paper) | {k} | {defrcn['map50']:.1f} | {defrcn['map50_95']:.1f} | - |")
        
        # YOLOv12 results
        map50_mean = averaged.loc[k, ('map50', 'mean')] * 100
        map50_std = averaged.loc[k, ('map50', 'std')] * 100
        map50_95_mean = averaged.loc[k, ('map50_95', 'mean')] * 100
        map50_95_std = averaged.loc[k, ('map50_95', 'std')] * 100
        n = int(averaged.loc[k, ('map50', 'count')])
        
        print(f"| YOLOv12 (ours) | {k} | **{map50_mean:.1f} ± {map50_std:.1f}** | **{map50_95_mean:.1f} ± {map50_95_std:.1f}** | {n} |")
    
    print("")
    print("="*80)
    
    # Detailed stats
    print("\nDetailed Statistics:")
    print("-" * 80)
    for k in sorted(averaged.index):
        print(f"\n{k}-shot:")
        print(f"  mAP50:    {averaged.loc[k, ('map50', 'mean')]*100:.2f} ± {averaged.loc[k, ('map50', 'std')]*100:.2f}")
        print(f"  mAP50-95: {averaged.loc[k, ('map50_95', 'mean')]*100:.2f} ± {averaged.loc[k, ('map50_95', 'std')]*100:.2f}")
        print(f"  Precision: {averaged.loc[k, ('precision', 'mean')]*100:.2f} ± {averaged.loc[k, ('precision', 'std')]*100:.2f}")
        print(f"  Recall:    {averaged.loc[k, ('recall', 'mean')]*100:.2f} ± {averaged.loc[k, ('recall', 'std')]*100:.2f}")
        print(f"  N seeds:   {int(averaged.loc[k, ('map50', 'count')])}")


def save_results(df_results, averaged, output_dir, protocol):
    """Save results to CSV files."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw results
    raw_file = output_dir / f'{protocol}_raw_results.csv'
    df_results.to_csv(raw_file, index=False)
    print(f"\n✅ Raw results saved to: {raw_file}")
    
    # Save averaged results
    avg_file = output_dir / f'{protocol}_averaged_results.csv'
    averaged.to_csv(avg_file)
    print(f"✅ Averaged results saved to: {avg_file}")


def main():
    parser = argparse.ArgumentParser(description='Average few-shot results across seeds')
    parser.add_argument('--protocol', type=str, required=True,
                        choices=['paper', 'modern'],
                        help='Protocol type')
    parser.add_argument('--base_dir', type=str, required=True,
                        help='Base directory containing seed results')
    parser.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5],
                        help='Seed values to average')
    parser.add_argument('--k_shots', type=int, nargs='+', default=[1, 3, 5, 10],
                        help='K-shot values')
    parser.add_argument('--output_dir', type=str,
                        default='/content/drive/MyDrive/yolov12_runs/results',
                        help='Output directory for saved results')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"Averaging Results for {args.protocol.upper()} Protocol")
    print(f"{'='*80}")
    print(f"Base directory: {args.base_dir}")
    print(f"Seeds: {args.seeds}")
    print(f"K-shots: {args.k_shots}")
    
    # Collect results
    print(f"\nCollecting results...")
    df_results = collect_results(args.base_dir, args.seeds, args.k_shots, args.protocol)
    
    if df_results.empty:
        print("❌ No results found!")
        return
    
    print(f"✅ Collected {len(df_results)} results")
    
    # Average across seeds
    print(f"\nAveraging across seeds...")
    averaged = average_results(df_results)
    
    # Print comparison table
    print_comparison_table(averaged, args.protocol)
    
    # Save results
    save_results(df_results, averaged, args.output_dir, args.protocol)
    
    print(f"\n{'='*80}")
    print(f"✅ Results averaging complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()

