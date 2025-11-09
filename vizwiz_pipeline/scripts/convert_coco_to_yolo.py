"""
Convert COCO Format to YOLO Format
===================================

Converts COCO format annotations to YOLO format.
Works with split files created by split_dataset.py.

Usage:
    python scripts/convert_coco_to_yolo.py \
        --coco_json /content/drive/MyDrive/datasets/nonprivate/base_images/instances_default.json \
        --split_file /content/splits/nonprivate/train.txt \
        --output_dir /content/drive/MyDrive/datasets/nonprivate/yolo_format/train \
        --split_name train
"""

import argparse
import json
from pathlib import Path
from tqdm import tqdm
from pycocotools.coco import COCO
import shutil


def convert_coco_to_yolo(coco_json_path, split_file_path, output_dir, split_name='train'):
    """
    Convert COCO format to YOLO format for a specific split.
    
    Args:
        coco_json_path: Path to COCO format JSON file
        split_file_path: Path to split file (txt file with image paths)
        output_dir: Output directory for YOLO format
        split_name: Name of split (train/val/test)
    """
    # Load COCO annotations
    print(f"Loading COCO annotations from: {coco_json_path}")
    coco = COCO(coco_json_path)
    
    # Create output directories
    images_dir = Path(output_dir) / "images"
    labels_dir = Path(output_dir) / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get category mapping (COCO category_id -> YOLO class_id)
    cat_ids = sorted(coco.getCatIds())
    cat_id_to_class = {cat_id: idx for idx, cat_id in enumerate(cat_ids)}
    
    # Load split file
    print(f"Loading split file: {split_file_path}")
    with open(split_file_path, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    print(f"Converting {len(image_paths)} images for {split_name}...")
    
    converted_count = 0
    skipped_count = 0
    
    # Create mapping from file_name to image_id
    file_name_to_img_id = {}
    for img_id in coco.getImgIds():
        img_info = coco.loadImgs([img_id])[0]
        file_name = img_info['file_name']
        # Handle different path formats
        file_name_to_img_id[file_name] = img_id
        file_name_to_img_id[Path(file_name).name] = img_id
    
    for img_path_str in tqdm(image_paths, desc=f"Processing {split_name}"):
        img_path = Path(img_path_str)
        
        if not img_path.exists():
            # Try to find the image
            img_name = img_path.name
            found = False
            for ext in ['.jpg', '.jpeg', '.png', '.jpg', '.jpeg']:
                alt_path = img_path.parent / f"{img_path.stem}{ext}"
                if alt_path.exists():
                    img_path = alt_path
                    found = True
                    break
            
            if not found:
                print(f"Warning: Image not found: {img_path_str}")
                skipped_count += 1
                continue
        
        # Get image ID from COCO data
        img_name = img_path.name
        img_id = None
        
        # Try to find image ID by file name
        if img_name in file_name_to_img_id:
            img_id = file_name_to_img_id[img_name]
        else:
            # Try without extension
            img_name_no_ext = img_path.stem
            for name, id_val in file_name_to_img_id.items():
                if Path(name).stem == img_name_no_ext:
                    img_id = id_val
                    break
        
        if img_id is None:
            print(f"Warning: Image ID not found for: {img_name}")
            skipped_count += 1
            continue
        
        # Load image info
        img_info = coco.loadImgs([img_id])[0]
        img_width = img_info['width']
        img_height = img_info['height']
        
        # Copy image to output directory
        dst_img = images_dir / img_path.name
        shutil.copy2(img_path, dst_img)
        
        # Get annotations for this image
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        anns = coco.loadAnns(ann_ids)
        
        # Create YOLO format label file
        label_filename = img_path.stem + '.txt'
        label_path = labels_dir / label_filename
        
        with open(label_path, 'w') as f:
            for ann in anns:
                if 'bbox' not in ann:
                    continue
                
                # COCO bbox format: [x, y, width, height] (top-left corner)
                x, y, w, h = ann['bbox']
                
                # Skip invalid boxes
                if w <= 1 or h <= 1:
                    continue
                
                # Convert to YOLO format: normalized center_x, center_y, width, height
                center_x = (x + w / 2) / img_width
                center_y = (y + h / 2) / img_height
                norm_w = w / img_width
                norm_h = h / img_height
                
                # Clamp values to [0, 1]
                center_x = max(0, min(1, center_x))
                center_y = max(0, min(1, center_y))
                norm_w = max(0, min(1, norm_w))
                norm_h = max(0, min(1, norm_h))
                
                # Get class ID (YOLO format uses 0-indexed classes)
                coco_cat_id = ann['category_id']
                yolo_class_id = cat_id_to_class[coco_cat_id]
                
                # Write to label file: class_id center_x center_y width height
                f.write(f"{yolo_class_id} {center_x:.6f} {center_y:.6f} {norm_w:.6f} {norm_h:.6f}\n")
        
        converted_count += 1
    
    print(f"\nConversion complete!")
    print(f"Converted: {converted_count} images")
    print(f"Skipped: {skipped_count} images")
    
    # Return category info for YAML generation
    cats = coco.loadCats(cat_ids)
    class_names = [cat['name'] for cat in sorted(cats, key=lambda x: x['id'])]
    
    return cat_ids, cat_id_to_class, class_names


def main():
    parser = argparse.ArgumentParser(description='Convert COCO format to YOLO format')
    parser.add_argument('--coco_json', type=str, required=True,
                        help='Path to COCO format JSON file')
    parser.add_argument('--split_file', type=str, required=True,
                        help='Path to split file (txt file with image paths)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for YOLO format')
    parser.add_argument('--split_name', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Name of split (default: train)')
    
    args = parser.parse_args()
    
    convert_coco_to_yolo(
        args.coco_json,
        args.split_file,
        args.output_dir,
        args.split_name
    )


if __name__ == '__main__':
    main()

