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
import re
from pathlib import Path
from collections import defaultdict
import random


def load_coco_json(coco_json_path):
    """Load COCO format JSON file."""
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    return coco_data


def get_base_stem(filename):
    """
    Extract base stem from filename by removing augmentation suffixes.
    
    Examples:
        '123_left_90.jpg' -> '123'
        '456_right_40.jpg' -> '456'
        '789.jpg' -> '789'
    """
    stem = Path(filename).stem
    # Remove patterns like _left_XX, _right_XX
    stem = re.sub(r'_(left|right)_\d+$', '', stem)
    return stem


def is_augmented_image(filename):
    """Check if filename represents an augmented image (left/right rotation)."""
    return bool(re.search(r'_(left|right)_\d+', Path(filename).stem))


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


def split_by_groups(coco_data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Split dataset by grouping variants of the same base image together.
    All augmented versions (left/right rotations) of an image go to the same split.
    
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
    
    # Group images by base stem
    base_stem_to_ids = defaultdict(list)
    for img in coco_data['images']:
        base_stem = get_base_stem(img['file_name'])
        base_stem_to_ids[base_stem].append(img['id'])
    
    # Split by groups
    base_stems = sorted(base_stem_to_ids.keys())
    random.seed(seed)
    random.shuffle(base_stems)
    
    # Calculate split indices
    n_total = len(base_stems)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # Assign groups to splits
    train_stems = set(base_stems[:n_train])
    val_stems = set(base_stems[n_train:n_train + n_val])
    test_stems = set(base_stems[n_train + n_val:])
    
    # Collect all image IDs for each split
    train_ids = set()
    val_ids = set()
    test_ids = set()
    
    for base_stem, img_ids in base_stem_to_ids.items():
        if base_stem in train_stems:
            train_ids.update(img_ids)
        elif base_stem in val_stems:
            val_ids.update(img_ids)
        elif base_stem in test_stems:
            test_ids.update(img_ids)
    
    n_total_images = len(train_ids) + len(val_ids) + len(test_ids)
    print(f"Total base groups: {n_total}")
    print(f"Total images: {n_total_images}")
    print(f"Train: {len(train_ids)} images from {len(train_stems)} groups ({len(train_ids)/n_total_images*100:.1f}%)")
    print(f"Val: {len(val_ids)} images from {len(val_stems)} groups ({len(val_ids)/n_total_images*100:.1f}%)")
    print(f"Test: {len(test_ids)} images from {len(test_stems)} groups ({len(test_ids)/n_total_images*100:.1f}%)")
    
    return train_ids, val_ids, test_ids


def write_split_files(coco_data, train_ids, val_ids, test_ids, images_dir, output_dir, dataset_type,
                      original_ratio=0.6, emit_subsets=False):
    """
    Write split files (train.txt, val.txt, test.txt) with image paths.
    Also optionally balance original/augmented ratio and emit subset files.
    
    Args:
        coco_data: COCO format dictionary
        train_ids, val_ids, test_ids: Sets of image IDs
        images_dir: Base directory containing images
        output_dir: Directory to save split files
        dataset_type: 'nonprivate' or 'private'
        original_ratio: Target ratio of original images in training (0.5-0.7)
        emit_subsets: Whether to emit separate lists for originals/augmented
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create mapping from image ID to file path and metadata
    id_to_path = {}
    id_to_filename = {}
    images_dir = Path(images_dir)
    
    for img in coco_data['images']:
        img_id = img['id']
        file_name = img['file_name']
        id_to_filename[img_id] = file_name
        
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
                images_dir / 'augmented' / 'left_rotate' / Path(file_name).name,
                images_dir / 'augmented' / 'right_rotate' / Path(file_name).name,
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
    
    # Process each split
    splits = {
        'train': train_ids,
        'val': val_ids,
        'test': test_ids
    }
    
    for split_name, split_ids in splits.items():
        # Separate originals and augmented
        original_ids = []
        augmented_ids = []
        
        for img_id in split_ids:
            if img_id not in id_to_path:
                continue
            filename = id_to_filename.get(img_id, '')
            if is_augmented_image(filename):
                augmented_ids.append(img_id)
            else:
                original_ids.append(img_id)
        
        # Write main split file with optional balancing for train
        split_file = output_dir / f"{split_name}.txt"
        
        if split_name == 'train' and original_ratio is not None and len(original_ids) > 0 and len(augmented_ids) > 0:
            # Balance the ratio by duplicating entries (simpler than sampling)
            current_ratio = len(original_ids) / (len(original_ids) + len(augmented_ids))
            print(f"Train set current original ratio: {current_ratio:.2%}")
            
            if abs(current_ratio - original_ratio) > 0.05:
                # Calculate duplication factors
                target_orig = original_ratio
                target_aug = 1 - original_ratio
                
                # Scale to match target ratio
                if current_ratio < original_ratio:
                    # Need more originals
                    orig_factor = int((target_orig * len(augmented_ids)) / (target_aug * len(original_ids)))
                    aug_factor = 1
                else:
                    # Need more augmented
                    orig_factor = 1
                    aug_factor = int((target_aug * len(original_ids)) / (target_orig * len(augmented_ids)))
                
                orig_factor = max(1, orig_factor)
                aug_factor = max(1, aug_factor)
                
                print(f"Balancing: orig_factor={orig_factor}, aug_factor={aug_factor}")
                
                balanced_ids = original_ids * orig_factor + augmented_ids * aug_factor
                random.shuffle(balanced_ids)
                
                with open(split_file, 'w') as f:
                    for img_id in balanced_ids:
                        f.write(f"{id_to_path[img_id]}\n")
                
                print(f"Written balanced {split_file} with {len(balanced_ids)} entries ({len(set(balanced_ids))} unique)")
            else:
                # Ratio is close enough, write as-is
                with open(split_file, 'w') as f:
                    for img_id in sorted(original_ids + augmented_ids):
                        f.write(f"{id_to_path[img_id]}\n")
                print(f"Written {split_file} with {len(original_ids) + len(augmented_ids)} images")
        else:
            # Write normally for val/test or when no balancing
            all_ids = sorted(original_ids + augmented_ids)
            with open(split_file, 'w') as f:
                for img_id in all_ids:
                    f.write(f"{id_to_path[img_id]}\n")
            print(f"Written {split_file} with {len(all_ids)} images")
        
        # Emit subset files if requested
        if emit_subsets and (len(original_ids) > 0 or len(augmented_ids) > 0):
            if len(original_ids) > 0:
                orig_file = output_dir / f"{split_name}_originals.txt"
                with open(orig_file, 'w') as f:
                    for img_id in sorted(original_ids):
                        f.write(f"{id_to_path[img_id]}\n")
                print(f"  └─ {orig_file.name} with {len(original_ids)} original images")
            
            if len(augmented_ids) > 0:
                aug_file = output_dir / f"{split_name}_augmented.txt"
                with open(aug_file, 'w') as f:
                    for img_id in sorted(augmented_ids):
                        f.write(f"{id_to_path[img_id]}\n")
                print(f"  └─ {aug_file.name} with {len(augmented_ids)} augmented images")
    
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
    parser = argparse.ArgumentParser(description='Split COCO dataset into train/val/test with optional grouping and balancing')
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
    parser.add_argument('--grouped', action='store_true',
                        help='Group augmented variants together (all variants in same split)')
    parser.add_argument('--original_ratio', type=float, default=None,
                        help='Target ratio of original images in training (e.g., 0.6 for 60%%)')
    parser.add_argument('--emit_subsets', action='store_true',
                        help='Emit separate lists for originals/augmented evaluation')
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
    if args.grouped:
        print("Using grouped split (variants stay together)")
        train_ids, val_ids, test_ids = split_by_groups(
            coco_data,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed
        )
    else:
        print("Using random split by image IDs")
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
        args.dataset_type,
        original_ratio=args.original_ratio,
        emit_subsets=args.emit_subsets
    )
    
    print("Done!")


if __name__ == '__main__':
    main()

