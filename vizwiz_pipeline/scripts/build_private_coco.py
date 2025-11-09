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
        --left_rotate_dir /content/datasets/private/augmented/left_rotate \
        --right_rotate_dir /content/datasets/private/augmented/right_rotate \
        --shifted_json /content/datasets/private/augmented/instances_shifted_from_original.json \
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
    
    # Helper: map image_id -> image dict for quick lookups
    print("Indexing shifted images...")
    shifted_images_by_id = {img['id']: img for img in shifted_data.get('images', [])}
    shifted_annotations_by_image = defaultdict(list)
    for ann in shifted_data['annotations']:
        shifted_annotations_by_image[ann['image_id']].append(ann)
    shifted_image_ids = set(shifted_annotations_by_image.keys())

    # Build quick lookup of stems present in left/right dirs
    def build_stem_to_files_map(directory: Path):
        stem_to_files = defaultdict(list)
        for ext in ('*.jpg', '*.jpeg', '*.png'):
            for p in directory.glob(ext):
                stem_to_files[p.stem].append(p)
        return stem_to_files

    # Resolve original stem for an image_id using shifted or query metadata
    def image_id_to_stem(img_id: int):
        img_meta = shifted_images_by_id.get(img_id)
        if img_meta and 'file_name' in img_meta:
            return Path(img_meta['file_name']).stem
        # fall back to query images if ids overlap
        for img in query_data['images']:
            if img['id'] == img_id:
                return Path(img['file_name']).stem
        return None

    # Fetch dimensions from known metadata
    def image_id_to_dimensions(img_id: int):
        img_meta = shifted_images_by_id.get(img_id)
        if img_meta and 'width' in img_meta and 'height' in img_meta:
            return img_meta['width'], img_meta['height']
        for img in query_data['images']:
            if img['id'] == img_id:
                return img['width'], img['height']
        return 640, 480

    print("Processing left_rotate images...")
    left_rotate_dir = Path(left_rotate_dir)
    left_stem_to_files = build_stem_to_files_map(left_rotate_dir)

    left_added = 0
    for original_img_id in shifted_image_ids:
        stem = image_id_to_stem(original_img_id)
        if not stem:
            continue
        files = left_stem_to_files.get(stem, [])
        if not files:
            continue
        # For each matching file (in case there are multiple versions), add an image and its annotations
        for matched_file in files:
            max_image_id += 1
            new_img_id = max_image_id
            w, h = image_id_to_dimensions(original_img_id)
            merged['images'].append({
                'id': new_img_id,
                'width': w,
                'height': h,
                'file_name': f"left_rotate/{matched_file.name}"
            })
            # Remap annotations from shifted to this new image
            for ann in shifted_annotations_by_image[original_img_id]:
                max_ann_id += 1
                new_ann = ann.copy()
                new_ann['id'] = max_ann_id
                new_ann['image_id'] = new_img_id
                merged['annotations'].append(new_ann)
            left_added += 1

    print("Processing right_rotate images...")
    right_rotate_dir = Path(right_rotate_dir)
    right_stem_to_files = build_stem_to_files_map(right_rotate_dir)

    right_added = 0
    for original_img_id in shifted_image_ids:
        stem = image_id_to_stem(original_img_id)
        if not stem:
            continue
        files = right_stem_to_files.get(stem, [])
        if not files:
            continue
        for matched_file in files:
            max_image_id += 1
            new_img_id = max_image_id
            w, h = image_id_to_dimensions(original_img_id)
            merged['images'].append({
                'id': new_img_id,
                'width': w,
                'height': h,
                'file_name': f"right_rotate/{matched_file.name}"
            })
            for ann in shifted_annotations_by_image[original_img_id]:
                max_ann_id += 1
                new_ann = ann.copy()
                new_ann['id'] = max_ann_id
                new_ann['image_id'] = new_img_id
                merged['annotations'].append(new_ann)
            right_added += 1
    
    # Save merged JSON
    output_path = Path(output_json_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"\nMerged dataset saved to: {output_path}")
    print(f"Total images: {len(merged['images'])}")
    print(f"  - query_images: {len(query_image_map)}")
    print(f"  - left_rotate added: {left_added}")
    print(f"  - right_rotate added: {right_added}")
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

