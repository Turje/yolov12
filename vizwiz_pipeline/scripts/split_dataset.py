"""
Dataset Splitting Script
========================

Splits COCO format datasets into 80/10/10 train/val/test splits.
Creates text files listing image paths for each split.

Usage:
    python scripts/split_dataset.py \
        --coco_json /content/drive/MyDrive/datasets/nonprivate/base_images/instances_default.json \
        --images_dir /content/drive/MyDrive/datasets/nonprivate/base_images \
        --output_dir /content/splits/nonprivate \
        --dataset_type nonprivate

    python scripts/split_dataset.py \
        --coco_json /content/drive/MyDrive/datasets/private/merged_private.json \
        --images_dir /content/drive/MyDrive/datasets/private \
        --output_dir /content/splits/private \
        --dataset_type private
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict
import random


def load_coco_json(coco_json_path):
    """Load COCO format JSON file."""
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data


def split_by_image_ids(coco_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split dataset by image IDs into train/val/test.
    
    Args:
        coco_data: COCO format dictionary
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility
    
    Returns:
        train_ids, val_ids, test_ids: Sets of image IDs
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Get all unique image IDs
    image_ids = sorted(set(img['id'] for img in coco_data['images']))
    
    # Set random seed
    random.seed(seed)
    random.shuffle(image_ids)
    
    # Calculate split indices
    n_total = len(image_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Split
    train_ids = set(image_ids[:n_train])
    val_ids = set(image_ids[n_train:n_train + n_val])
    test_ids = set(image_ids[n_train + n_val:])
    
    print(f"Total images: {n_total}")
    print(f"Train: {len(train_ids)} ({len(train_ids)/n_total*100:.1f}%)")
    print(f"Val: {len(val_ids)} ({len(val_ids)/n_total*100:.1f}%)")
    print(f"Test: {len(test_ids)} ({len(test_ids)/n_total*100:.1f}%)")
    
    return train_ids, val_ids, test_ids


def write_split_files(coco_data, train_ids, val_ids, test_ids, images_dir, output_dir, dataset_type):
    """
    Write split files (train.txt, val.txt, test.txt) with image paths.
    
    Args:
        coco_data: COCO format dictionary
        train_ids, val_ids, test_ids: Sets of image IDs
        images_dir: Base directory containing images
        output_dir: Directory to save split files
        dataset_type: 'nonprivate' or 'private'
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mapping from image ID to file path
    id_to_path = {}
    images_dir = Path(images_dir)
    
    for img in coco_data['images']:
        img_id = img['id']
        file_name = img['file_name']
        
        # Handle different possible image directory structures
        # For nonprivate: images might be directly in base_images/ or in base_images/images/
        # For private: images might be in query_images/, left_rotate/, right_rotate/
        if dataset_type == 'nonprivate':
            # Try common structures
            possible_paths = [
                images_dir / file_name,
                images_dir / 'images' / file_name,
            ]
        else:  # private
            # Private dataset has images in multiple folders
            # The merged JSON should have paths relative to the private root
            possible_paths = [
                images_dir / file_name,
                images_dir / 'query_images' / file_name,
                images_dir / 'left_rotate' / file_name,
                images_dir / 'right_rotate' / file_name,
            ]
        
        # Find the first existing path
        img_path = None
        for path in possible_paths:
            if path.exists():
                img_path = path
                break
        
        if img_path is None:
            print(f"Warning: Image not found: {file_name}")
            continue
        
        id_to_path[img_id] = str(img_path)
    
    # Write split files
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    for split_name, split_ids in splits.items():
        split_file = output_dir / f"{split_name}.txt"
        with open(split_file, 'w') as f:
            for img_id in sorted(split_ids):
                if img_id in id_to_path:
                    f.write(f"{id_to_path[img_id]}\n")
        
        print(f"Written {split_file} with {len([id for id in split_ids if id in id_to_path])} images")
    
    # Also save the split IDs as JSON for reference
    split_info = {
        'train_ids': sorted(list(train_ids)),
        'val_ids': sorted(list(val_ids)),
        'test_ids': sorted(list(test_ids)),
    }
    with open(output_dir / 'split_info.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Split info saved to {output_dir / 'split_info.json'}")


def main():
    parser = argparse.ArgumentParser(description='Split COCO dataset into train/val/test')
    parser.add_argument('--coco_json', type=str, required=True,
                        help='Path to COCO format JSON file')
    parser.add_argument('--images_dir', type=str, required=True,
                        help='Base directory containing images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for split files')
    parser.add_argument('--dataset_type', type=str, required=True,
                        choices=['nonprivate', 'private'],
                        help='Dataset type (nonprivate or private)')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                        help='Training set ratio (default: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                        help='Validation set ratio (default: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Test set ratio (default: 0.1)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    print(f"Loading COCO JSON from: {args.coco_json}")
    coco_data = load_coco_json(args.coco_json)
    
    print(f"Splitting dataset...")
    train_ids, val_ids, test_ids = split_by_image_ids(
        coco_data,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    print(f"Writing split files to: {args.output_dir}")
    write_split_files(
        coco_data,
        train_ids,
        val_ids,
        test_ids,
        args.images_dir,
        args.output_dir,
        args.dataset_type
    )
    
    print("Done!")


if __name__ == '__main__':
    main()

