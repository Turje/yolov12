"""
Build Merged Private COCO Dataset
==================================

Merges COCO annotations from:
- query_images/instances_default.json
- left_rotate/instances_shifted_from_original.json
- right_rotate/instances_shifted_from_original.json

Into a single merged COCO format file.

Usage:
    python scripts/build_private_coco.py \
        --query_json /content/datasets/private/query_images/instances_default.json \
        --query_images_dir /content/datasets/private/query_images \
        --left_rotate_dir /content/datasets/private/left_rotate \
        --right_rotate_dir /content/datasets/private/right_rotate \
        --shifted_json /content/datasets/private/left_rotate/instances_shifted_from_original.json \
        --output_json /content/datasets/private/merged_private.json
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict


def load_coco_json(json_path):
    """Load COCO format JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def merge_private_coco(query_json_path, query_images_dir, left_rotate_dir, right_rotate_dir,
                       shifted_json_path, output_json_path):
    """
    Merge private dataset COCO annotations.
    
    Args:
        query_json_path: Path to query_images/instances_default.json
        query_images_dir: Directory containing query images
        left_rotate_dir: Directory containing left_rotate images
        right_rotate_dir: Directory containing right_rotate images
        shifted_json_path: Path to instances_shifted_from_original.json
        output_json_path: Path to save merged JSON
    """
    print("Loading query images annotations...")
    query_data = load_coco_json(query_json_path)
    
    print("Loading shifted annotations...")
    shifted_data = load_coco_json(shifted_json_path)
    
    # Initialize merged structure
    merged = {
        'info': query_data.get('info', {}),
        'licenses': query_data.get('licenses', []),
        'categories': query_data['categories'],  # Use categories from query
        'images': [],
        'annotations': []
    }
    
    # Track IDs to avoid conflicts
    max_image_id = 0
    max_ann_id = 0
    
    # Process query images
    print("Processing query images...")
    query_images_dir = Path(query_images_dir)
    query_image_map = {}  # Map original image IDs to new IDs
    
    for img in query_data['images']:
        original_id = img['id']
        max_image_id = max(max_image_id, original_id)
        
        # Update file path if needed
        file_name = img['file_name']
        if not (query_images_dir / file_name).exists():
            # Try to find the image
            found = False
            for ext in ['.jpg', '.jpeg', '.png']:
                if (query_images_dir / f"{Path(file_name).stem}{ext}").exists():
                    file_name = f"{Path(file_name).stem}{ext}"
                    found = True
                    break
            if not found:
                print(f"Warning: Query image not found: {file_name}")
                continue
        
        new_img = img.copy()
        new_img['id'] = original_id
        new_img['file_name'] = f"query_images/{file_name}"
        merged['images'].append(new_img)
        query_image_map[original_id] = original_id
    
    # Process query annotations
    for ann in query_data['annotations']:
        if ann['image_id'] in query_image_map:
            new_ann = ann.copy()
            new_ann['id'] = ann['id']
            max_ann_id = max(max_ann_id, ann['id'])
            merged['annotations'].append(new_ann)
    
    # Process left_rotate images from shifted JSON
    print("Processing left_rotate images...")
    left_rotate_dir = Path(left_rotate_dir)
    left_image_map = {}  # Map original image IDs to new image IDs
    
    # Find images in left_rotate that correspond to annotations in shifted JSON
    shifted_image_ids = set(ann['image_id'] for ann in shifted_data['annotations'])
    
    # Group annotations by image_id
    left_annotations_by_image = defaultdict(list)
    for ann in shifted_data['annotations']:
        left_annotations_by_image[ann['image_id']].append(ann)
    
    # Find corresponding images in left_rotate directory
    for original_img_id in shifted_image_ids:
        # Try to find the image file
        # The image might have the same name as in query, or a modified name
        found_image = None
        for img_file in left_rotate_dir.glob('*.jpg'):
            found_image = img_file
            break
        for img_file in left_rotate_dir.glob('*.jpeg'):
            if found_image is None:
                found_image = img_file
            break
        
        if found_image is None:
            print(f"Warning: No image found in left_rotate for image_id {original_img_id}")
            continue
        
        # Create new image entry
        max_image_id += 1
        new_img_id = max_image_id
        
        # Try to get image dimensions from original or use defaults
        img_width = 640
        img_height = 480
        if original_img_id in query_image_map:
            orig_img = next((img for img in query_data['images'] if img['id'] == original_img_id), None)
            if orig_img:
                img_width = orig_img['width']
                img_height = orig_img['height']
        
        new_img = {
            'id': new_img_id,
            'width': img_width,
            'height': img_height,
            'file_name': f"left_rotate/{found_image.name}"
        }
        merged['images'].append(new_img)
        left_image_map[original_img_id] = new_img_id
    
    # Process left_rotate annotations
    for ann in shifted_data['annotations']:
        if ann['image_id'] in left_image_map:
            max_ann_id += 1
            new_ann = ann.copy()
            new_ann['id'] = max_ann_id
            new_ann['image_id'] = left_image_map[ann['image_id']]
            merged['annotations'].append(new_ann)
    
    # Process right_rotate images (similar to left_rotate)
    print("Processing right_rotate images...")
    right_rotate_dir = Path(right_rotate_dir)
    right_image_map = {}
    
    # Use the same shifted JSON but map to right_rotate images
    for original_img_id in shifted_image_ids:
        found_image = None
        for img_file in right_rotate_dir.glob('*.jpg'):
            found_image = img_file
            break
        for img_file in right_rotate_dir.glob('*.jpeg'):
            if found_image is None:
                found_image = img_file
            break
        
        if found_image is None:
            print(f"Warning: No image found in right_rotate for image_id {original_img_id}")
            continue
        
        max_image_id += 1
        new_img_id = max_image_id
        
        img_width = 640
        img_height = 480
        if original_img_id in query_image_map:
            orig_img = next((img for img in query_data['images'] if img['id'] == original_img_id), None)
            if orig_img:
                img_width = orig_img['width']
                img_height = orig_img['height']
        
        new_img = {
            'id': new_img_id,
            'width': img_width,
            'height': img_height,
            'file_name': f"right_rotate/{found_image.name}"
        }
        merged['images'].append(new_img)
        right_image_map[original_img_id] = new_img_id
    
    # Process right_rotate annotations
    for ann in shifted_data['annotations']:
        if ann['image_id'] in right_image_map:
            max_ann_id += 1
            new_ann = ann.copy()
            new_ann['id'] = max_ann_id
            new_ann['image_id'] = right_image_map[ann['image_id']]
            merged['annotations'].append(new_ann)
    
    # Save merged JSON
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"\nMerged dataset saved to: {output_path}")
    print(f"Total images: {len(merged['images'])}")
    print(f"Total annotations: {len(merged['annotations'])}")
    print(f"Categories: {len(merged['categories'])}")


def main():
    parser = argparse.ArgumentParser(description='Merge private dataset COCO annotations')
    parser.add_argument('--query_json', type=str, required=True,
                        help='Path to query_images/instances_default.json')
    parser.add_argument('--query_images_dir', type=str, required=True,
                        help='Directory containing query images')
    parser.add_argument('--left_rotate_dir', type=str, required=True,
                        help='Directory containing left_rotate images')
    parser.add_argument('--right_rotate_dir', type=str, required=True,
                        help='Directory containing right_rotate images')
    parser.add_argument('--shifted_json', type=str, required=True,
                        help='Path to instances_shifted_from_original.json')
    parser.add_argument('--output_json', type=str, required=True,
                        help='Path to save merged JSON file')
    
    args = parser.parse_args()
    
    merge_private_coco(
        args.query_json,
        args.query_images_dir,
        args.left_rotate_dir,
        args.right_rotate_dir,
        args.shifted_json,
        args.output_json
    )
    
    print("Done!")


if __name__ == '__main__':
    main()

