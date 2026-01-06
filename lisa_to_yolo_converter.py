"""
Convert LISA Traffic Light Dataset to YOLO format
Creates the directory structure and annotation files needed for YOLOv8
"""

import pandas as pd
import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Configuration
LISA_ROOT = "LISA Traffic Light Dataset"
ANNOTATIONS_ROOT = os.path.join(LISA_ROOT, "Annotations", "Annotations")
OUTPUT_ROOT = "LISA_YOLO"

# YOLO class mapping (must match exact order)
CLASS_NAMES = ['go', 'goLeft', 'stop', 'stopLeft', 'warning', 'warningLeft']
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}

# Class name normalization (same as before)
CLASS_MAPPING = {
    'go': 'go',
    'goLeft': 'goLeft',
    'go_forward': 'go',
    'go_traffic_light': 'go',
    'goLeft_traffic_light': 'goLeft',
    'go_forward_traffic_light': 'go',
    'stop': 'stop',
    'stopLeft': 'stopLeft',
    'stop_traffic_light': 'stop',
    'stopLeft_traffic_light': 'stopLeft',
    'warning': 'warning',
    'warningLeft': 'warningLeft',
    'warning_traffic_light': 'warning',
    'warningLeft_traffic_light': 'warningLeft',
}


def normalize_class_name(raw_class):
    """Normalize class name to base class"""
    clean = str(raw_class).strip().lower().replace(' ', '').replace('-', '')
    
    for original, mapped in CLASS_MAPPING.items():
        if clean == original.lower().replace('_', ''):
            return mapped
    
    # Fallback matching
    if 'goleft' in clean or 'leftgo' in clean:
        return 'goLeft'
    elif 'go' in clean or 'green' in clean:
        return 'go'
    elif 'stopleft' in clean or 'leftstop' in clean:
        return 'stopLeft'
    elif 'stop' in clean or 'red' in clean:
        return 'stop'
    elif 'warningleft' in clean or 'leftwarning' in clean:
        return 'warningLeft'
    elif 'warning' in clean or 'yellow' in clean:
        return 'warning'
    
    return None


def parse_annotations(csv_path):
    """Parse LISA CSV annotations"""
    df = pd.read_csv(csv_path, delimiter=';')
    grouped = {}
    
    for _, row in df.iterrows():
        filename = row['Filename']
        
        # Fix filename path
        if '/' in filename:
            filename = filename.split('/')[-1]
        if '\\' in filename:
            filename = filename.split('\\')[-1]
        
        if filename not in grouped:
            grouped[filename] = []
        
        # Normalize class
        raw_tag = str(row['Annotation tag']).strip()
        matched_label = None
        
        for cname in CLASS_NAMES:
            if raw_tag.lower().replace(' ', '').replace('-', '') == cname.lower().replace(' ', '').replace('-', ''):
                matched_label = cname
                break
        
        if matched_label is None:
            matched_label = normalize_class_name(raw_tag)
        
        if matched_label is None or matched_label not in CLASS_TO_IDX:
            continue
        
        annotation = {
            'x1': int(row['Upper left corner X']),
            'y1': int(row['Upper left corner Y']),
            'x2': int(row['Lower right corner X']),
            'y2': int(row['Lower right corner Y']),
            'class': matched_label
        }
        grouped[filename].append(annotation)
    
    return grouped


def convert_to_yolo_format(annotation, img_width, img_height):
    """
    Convert LISA annotation to YOLO format
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All values normalized to [0, 1]
    """
    x1, y1 = annotation['x1'], annotation['y1']
    x2, y2 = annotation['x2'], annotation['y2']
    
    # Calculate center and dimensions (normalized)
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    class_id = CLASS_TO_IDX[annotation['class']]
    
    return f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"


def process_clip(clip_path, img_dir, output_split, split_name):
    """Process a single video clip"""
    bulb_csv = os.path.join(clip_path, "frameAnnotationsBULB.csv")
    
    if not os.path.exists(bulb_csv):
        return 0, 0
    
    annotations = parse_annotations(bulb_csv)
    
    images_copied = 0
    labels_created = 0
    
    for filename, boxes in annotations.items():
        img_path = os.path.join(img_dir, filename)
        
        if not os.path.exists(img_path):
            continue
        
        # Get image dimensions
        import cv2
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        
        # Copy image to YOLO directory
        output_img_path = os.path.join(OUTPUT_ROOT, output_split, 'images', filename)
        os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
        shutil.copy2(img_path, output_img_path)
        images_copied += 1
        
        # Create YOLO annotation file
        label_filename = filename.replace('.jpg', '.txt').replace('.png', '.txt')
        output_label_path = os.path.join(OUTPUT_ROOT, output_split, 'labels', label_filename)
        os.makedirs(os.path.dirname(output_label_path), exist_ok=True)
        
        with open(output_label_path, 'w') as f:
            for box in boxes:
                yolo_line = convert_to_yolo_format(box, img_width, img_height)
                f.write(yolo_line + '\n')
        labels_created += 1
    
    return images_copied, labels_created


def main():
    """Main conversion function"""
    print("="*80)
    print("LISA → YOLO Format Converter")
    print("="*80)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(OUTPUT_ROOT, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(OUTPUT_ROOT, split, 'labels'), exist_ok=True)
    
    total_images = 0
    total_labels = 0
    
    # ========================================
    # Process TRAINING data (dayTrain + nightTrain)
    # ========================================
    print("\n[1/3] Processing training data...")
    
    # Day training clips
    day_train_path = os.path.join(ANNOTATIONS_ROOT, "dayTrain")
    if os.path.exists(day_train_path):
        clips = [c for c in os.listdir(day_train_path) if os.path.isdir(os.path.join(day_train_path, c))]
        
        # 85% train, 15% val split
        split_idx = int(0.85 * len(clips))
        train_clips = clips[:split_idx]
        val_clips = clips[split_idx:]
        
        print(f"  Day clips - Train: {len(train_clips)}, Val: {len(val_clips)}")
        
        # Process train clips
        for clip in tqdm(train_clips, desc="  Day train clips"):
            clip_path = os.path.join(day_train_path, clip)
            img_dir = os.path.join(LISA_ROOT, "dayTrain", "dayTrain", clip, "frames")
            imgs, lbls = process_clip(clip_path, img_dir, 'train', f'dayTrain/{clip}')
            total_images += imgs
            total_labels += lbls
        
        # Process val clips
        for clip in tqdm(val_clips, desc="  Day val clips"):
            clip_path = os.path.join(day_train_path, clip)
            img_dir = os.path.join(LISA_ROOT, "dayTrain", "dayTrain", clip, "frames")
            imgs, lbls = process_clip(clip_path, img_dir, 'val', f'dayTrain/{clip}')
            total_images += imgs
            total_labels += lbls
    
    # Night training clips
    night_train_path = os.path.join(ANNOTATIONS_ROOT, "nightTrain")
    if os.path.exists(night_train_path):
        clips = [c for c in os.listdir(night_train_path) if os.path.isdir(os.path.join(night_train_path, c))]
        
        split_idx = int(0.85 * len(clips))
        train_clips = clips[:split_idx]
        val_clips = clips[split_idx:]
        
        print(f"  Night clips - Train: {len(train_clips)}, Val: {len(val_clips)}")
        
        for clip in tqdm(train_clips, desc="  Night train clips"):
            clip_path = os.path.join(night_train_path, clip)
            img_dir = os.path.join(LISA_ROOT, "nightTrain", "nightTrain", clip, "frames")
            imgs, lbls = process_clip(clip_path, img_dir, 'train', f'nightTrain/{clip}')
            total_images += imgs
            total_labels += lbls
        
        for clip in tqdm(val_clips, desc="  Night val clips"):
            clip_path = os.path.join(night_train_path, clip)
            img_dir = os.path.join(LISA_ROOT, "nightTrain", "nightTrain", clip, "frames")
            imgs, lbls = process_clip(clip_path, img_dir, 'val', f'nightTrain/{clip}')
            total_images += imgs
            total_labels += lbls
    
    # ========================================
    # Process TEST data
    # ========================================
    print("\n[2/3] Processing test data...")
    
    test_sequences = ['daySequence1', 'daySequence2', 'nightSequence1', 
                     'nightSequence2', 'sample-dayClip6', 'sample-nightClip1']
    
    for seq in tqdm(test_sequences, desc="  Test sequences"):
        seq_path = os.path.join(ANNOTATIONS_ROOT, seq)
        if os.path.exists(seq_path):
            img_dir = os.path.join(LISA_ROOT, seq, seq, "frames")
            imgs, lbls = process_clip(seq_path, img_dir, 'test', seq)
            total_images += imgs
            total_labels += lbls
    
    # ========================================
    # Create YAML config file
    # ========================================
    print("\n[3/3] Creating YAML config...")
    
    yaml_content = f"""# LISA Traffic Light Dataset for YOLOv8
path: {os.path.abspath(OUTPUT_ROOT)}
train: train/images
val: val/images
test: test/images

# Classes
names:
  0: go
  1: goLeft
  2: stop
  3: stopLeft
  4: warning
  5: warningLeft

# Number of classes
nc: 6
"""
    
    yaml_path = os.path.join(OUTPUT_ROOT, 'lisa.yaml')
    with open(yaml_path, 'w') as f:
        f.write(yaml_content)
    
    print(f"\nYAML config saved: {yaml_path}")
    
    # ========================================
    # Summary
    # ========================================
    print("\n" + "="*80)
    print("CONVERSION COMPLETE")
    print("="*80)
    print(f"Total images processed: {total_images}")
    print(f"Total labels created: {total_labels}")
    
    # Count files per split
    for split in ['train', 'val', 'test']:
        img_dir = os.path.join(OUTPUT_ROOT, split, 'images')
        lbl_dir = os.path.join(OUTPUT_ROOT, split, 'labels')
        n_imgs = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]) if os.path.exists(img_dir) else 0
        n_lbls = len([f for f in os.listdir(lbl_dir) if f.endswith('.txt')]) if os.path.exists(lbl_dir) else 0
        print(f"{split.capitalize():5s}: {n_imgs} images, {n_lbls} labels")
    
    print("\n✅ Ready for YOLOv8 training!")
    print(f"   Config file: {yaml_path}")
    print("="*80)


if __name__ == "__main__":
    main()
