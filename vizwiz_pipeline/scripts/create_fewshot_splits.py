#!/usr/bin/env python3
"""
Create K-shot splits for few-shot learning experiments.

This script samples K images per class from the private dataset
to create 1-shot, 3-shot, 5-shot, and 10-shot training sets.

Usage:
    python scripts/create_fewshot_splits.py \
        --input_json /content/datasets/private/merged_private.json \
        --output_dir /content/datasets/private_fewshot \
        --k_shots 1 3 5 10 \
        --seed 42
"""

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
import argparse


def get_base_stem(filename):
    """
    Extract base image stem to group augmented variants.
    
    Examples:
        '123.jpg' -> '123'
        'left_rotate/456_left_70.jpg' -> '456'
        'right_rotate/789_right_40.jpeg' -> '789'
    """
    stem = Path(filename).stem
    
    # Remove augmentation suffixes
    if '_left_' in stem:
        return stem.split('_left_')[0]
    elif '_right_' in stem:
        return stem.split('_right_')[0]
    else:
        return stem


def create_k_shot_split(coco_json, k, output_dir, seed=42, include_augmented=False):
    """
    Create a K-shot dataset: K images per class.
    
    Args:
        coco_json: Path to merged_private.json
        k: Number of shots per class
        output_dir: Where to save the K-shot COCO JSON
        seed: Random seed for reproducibility
        include_augmented: If True, can sample augmented images. If False, only originals.
    
    Returns:
        Path to created K-shot JSON file
    """
    random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Creating {k}-shot split")
    print(f"{'='*60}")
    
    with open(coco_json, 'r') as f:
        data = json.load(f)
    
    # Group images by category
    # Use primary category (first annotation) for each image
    img_id_to_anns = defaultdict(list)
    for ann in data['annotations']:
        img_id_to_anns[ann['image_id']].append(ann)
    
    cat_to_images = defaultdict(list)
    for img in data['images']:
        img_id = img['id']
        if img_id in img_id_to_anns:
            # Get primary category (first annotation)
            primary_cat = img_id_to_anns[img_id][0]['category_id']
            
            # Filter augmented images if requested
            if include_augmented:
                cat_to_images[primary_cat].append(img_id)
            else:
                # Only include non-augmented (original query) images
                filename = img['file_name']
                if 'left_rotate' not in filename and 'right_rotate' not in filename:
                    cat_to_images[primary_cat].append(img_id)
    
    # Sample K images per category
    selected_img_ids = set()
    sampling_report = []
    
    for cat_id in sorted(cat_to_images.keys()):
        img_ids = cat_to_images[cat_id]
        cat_name = next(c['name'] for c in data['categories'] if c['id'] == cat_id)
        
        if len(img_ids) < k:
            print(f"⚠️  Category {cat_id} ({cat_name}) has only {len(img_ids)} images (need {k})")
            selected_img_ids.update(img_ids)
            sampling_report.append({
                'category_id': cat_id,
                'category_name': cat_name,
                'available': len(img_ids),
                'sampled': len(img_ids)
            })
        else:
            sampled = random.sample(img_ids, k)
            selected_img_ids.update(sampled)
            sampling_report.append({
                'category_id': cat_id,
                'category_name': cat_name,
                'available': len(img_ids),
                'sampled': k
            })
    
    # Create new COCO JSON with selected images
    new_images = [img for img in data['images'] if img['id'] in selected_img_ids]
    new_annotations = [ann for ann in data['annotations'] if ann['image_id'] in selected_img_ids]
    
    new_data = {
        'images': new_images,
        'annotations': new_annotations,
        'categories': data['categories'],
        'info': {
            'description': f'{k}-shot private dataset for few-shot learning',
            'k_shot': k,
            'seed': seed,
            'include_augmented': include_augmented,
            'sampling_report': sampling_report
        }
    }
    
    # Save K-shot JSON
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / f'{k}shot_private.json'
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=2)
    
    # Print summary
    print(f"\n✅ Created {k}-shot dataset:")
    print(f"   Total images: {len(new_images)}")
    print(f"   Total annotations: {len(new_annotations)}")
    print(f"   Categories: {len(data['categories'])}")
    print(f"   Saved to: {output_path}")
    
    # Print per-category breakdown
    print(f"\n   Per-category sampling:")
    for report in sampling_report:
        status = "✓" if report['sampled'] == k else "⚠"
        print(f"   {status} {report['category_name']:30s}: {report['sampled']}/{report['available']} images")
    
    return output_path


def create_fewshot_yaml(k, output_dir, base_images_dir, nc=16, class_names=None):
    """
    Generate YAML configuration for K-shot dataset.
    
    Args:
        k: Number of shots
        output_dir: Where YAML will be saved
        base_images_dir: Root directory containing images
        nc: Number of classes
        class_names: List of class names (optional, will use defaults if None)
    """
    if class_names is None:
        class_names = [
            'local_newspaper', 'bank_statement', 'bills_or_receipt', 'business_card',
            'condom_box', 'credit_or_debit_card', 'doctors_prescription', 'letters_with_address',
            'medical_record_document', 'pregnancy_test', 'empty_pill_bottle', 'tattoo_sleeve',
            'transcript', 'mortgage_or_investment_report', 'condom_with_plastic_bag', 'pregnancy_test_box'
        ]
    
    yaml_content = f"""# {k}-shot Private Dataset Configuration
# Generated for few-shot learning experiments

# Paths to split files
train: {output_dir}/{k}shot_train.txt
val: {output_dir}/{k}shot_val.txt

# Number of classes
nc: {nc}

# Class names (0-indexed)
names:
"""
    
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"
    
    yaml_path = Path(output_dir) / f'private_{k}shot.yaml'
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"   ✅ Created YAML: {yaml_path}")
    
    return yaml_path


def create_split_files(k_shot_json, output_dir, base_images_dir, val_ratio=0.2, seed=42):
    """
    Create train.txt and val.txt for K-shot dataset.
    
    Args:
        k_shot_json: Path to K-shot COCO JSON
        output_dir: Where to save split files
        base_images_dir: Root directory containing images
        val_ratio: Fraction of images to use for validation
        seed: Random seed
    """
    random.seed(seed)
    
    with open(k_shot_json, 'r') as f:
        data = json.load(f)
    
    # Get all image IDs
    img_ids = [img['id'] for img in data['images']]
    
    # Shuffle and split
    random.shuffle(img_ids)
    split_idx = int(len(img_ids) * (1 - val_ratio))
    
    train_ids = set(img_ids[:split_idx])
    val_ids = set(img_ids[split_idx:])
    
    # Create image path mapping
    img_id_to_path = {img['id']: img['file_name'] for img in data['images']}
    
    # Write train.txt
    k = data['info']['k_shot']
    train_file = Path(output_dir) / f'{k}shot_train.txt'
    val_file = Path(output_dir) / f'{k}shot_val.txt'
    
    with open(train_file, 'w') as f:
        for img_id in train_ids:
            rel_path = img_id_to_path[img_id]
            abs_path = Path(base_images_dir) / rel_path
            f.write(f"{abs_path}\n")
    
    with open(val_file, 'w') as f:
        for img_id in val_ids:
            rel_path = img_id_to_path[img_id]
            abs_path = Path(base_images_dir) / rel_path
            f.write(f"{abs_path}\n")
    
    print(f"   ✅ Created split files:")
    print(f"      Train: {train_file} ({len(train_ids)} images)")
    print(f"      Val:   {val_file} ({len(val_ids)} images)")


def main():
    parser = argparse.ArgumentParser(description='Create K-shot splits for few-shot learning')
    parser.add_argument('--input_json', type=str, required=True,
                        help='Path to merged_private.json')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for K-shot datasets')
    parser.add_argument('--base_images_dir', type=str, required=True,
                        help='Root directory containing images (e.g., /content/datasets/private)')
    parser.add_argument('--k_shots', type=int, nargs='+', default=[1, 3, 5, 10],
                        help='K values to create (default: 1 3 5 10)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--include_augmented', action='store_true',
                        help='Include augmented (rotated) images in sampling')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Fraction of K-shot data to use for validation (default: 0.2)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*60}")
    print(f"Creating Few-Shot Splits")
    print(f"{'='*60}")
    print(f"Input JSON: {args.input_json}")
    print(f"Output directory: {args.output_dir}")
    print(f"K-shot values: {args.k_shots}")
    print(f"Random seed: {args.seed}")
    print(f"Include augmented: {args.include_augmented}")
    print(f"Validation ratio: {args.val_ratio}")
    
    # Load categories from input JSON
    with open(args.input_json, 'r') as f:
        data = json.load(f)
    class_names = [cat['name'] for cat in sorted(data['categories'], key=lambda x: x['id'])]
    nc = len(class_names)
    
    # Create K-shot splits
    for k in args.k_shots:
        # Create K-shot JSON
        k_shot_json = create_k_shot_split(
            args.input_json, 
            k, 
            args.output_dir,
            args.seed,
            args.include_augmented
        )
        
        # Create train/val split files
        create_split_files(
            k_shot_json,
            args.output_dir,
            args.base_images_dir,
            args.val_ratio,
            args.seed
        )
        
        # Create YAML config
        create_fewshot_yaml(
            k,
            args.output_dir,
            args.base_images_dir,
            nc,
            class_names
        )
    
    print(f"\n{'='*60}")
    print(f"✅ All K-shot splits created successfully!")
    print(f"{'='*60}")
    print(f"\nNext steps:")
    print(f"1. Train K-shot models:")
    for k in args.k_shots:
        print(f"   python scripts/train_fewshot.py --k_shot {k}")
    print(f"\n2. Or run all at once:")
    print(f"   python scripts/run_fewshot_experiments.py")


if __name__ == '__main__':
    main()

