"""
YOLOv8 Training Script for LISA Traffic Light Dataset - OPTIMIZED
Key improvements for better performance on small traffic light objects
"""

from ultralytics import YOLO
import torch
import yaml
from pathlib import Path
import os

# ============================================================================
# IMPROVED CONFIGURATION FOR TRAFFIC LIGHTS
# ============================================================================

# Model configuration - CONSTRAINED FOR ESP32-CAM
MODEL_SIZE = 'yolov8n'  # MUST use nano - ESP32 has only 8MB PSRAM
                        # We'll compensate with better training instead

# Training hyperparameters - OPTIMIZED FOR SMALL OBJECTS
CONFIG = {
    # === CRITICAL CHANGES FOR ESP32 CONSTRAINTS ===
    'epochs': 200,          # ← INCREASED - compensate for smaller model
    'imgsz': 320,           # ← KEEP 320 - ESP32 can't handle larger input
    'batch': 16,            # Keep 16 (good balance)
    'device': 0,        # Change to 0 if you have GPU
    
    # === TRAINING STABILITY ===
    'workers': 4,
    'patience': 25,         # ← INCREASED from 15 (don't give up too early)
    'save': True,
    'save_period': 10,
    'cache': True,          # ← CHANGED to True (faster training if you have RAM)
    'resume': False,
    'amp': True,
    'pretrained': True,
    
    # === OPTIMIZER (TUNED FOR NANO MODEL) ===
    'optimizer': 'AdamW',
    'lr0': 0.003,          # ← HIGHER LR - nano needs to learn efficiently
    'lrf': 0.0001,         # ← VERY LOW final LR for fine-tuning
    'momentum': 0.937,
    'weight_decay': 0.0005,
    'warmup_epochs': 10.0,  # ← LONGER warmup for stability
    'warmup_momentum': 0.8,
    
    # === LOSS WEIGHTS (CRITICAL FOR NANO + SMALL OBJECTS) ===
    'box': 7.5,            # Keep default
    'cls': 2.0,            # ← MUCH HIGHER - force nano to focus on classification
    'dfl': 1.5,            # Keep default
    
    # === DATA AUGMENTATION (AGGRESSIVE - COMPENSATE FOR SMALL MODEL) ===
    'hsv_h': 0.05,         # ← VERY HIGH - force robustness to color shifts
    'hsv_s': 0.9,          # ← VERY HIGH - extreme saturation changes
    'hsv_v': 0.6,          # ← HIGH - day/night/shadow variations
    'degrees': 10.0,       # ← INCREASED - more rotation tolerance
    'translate': 0.2,      # ← HIGH - position invariance
    'scale': 0.9,          # ← VERY HIGH - critical for small objects!
    'shear': 0.0,          # Keep 0 (not useful for traffic lights)
    'perspective': 0.0002, # ← SLIGHT perspective distortion
    'flipud': 0.0,         # Keep 0 (traffic lights don't flip vertically)
    'fliplr': 0.5,         # Keep 0.5 (horizontal flip is fine)
    'mosaic': 1.0,         # Keep 1.0 (essential for small objects)
    'mixup': 0.15,         # ← HIGHER - more hard negative mining
    'copy_paste': 0.15,    # ← HIGHER - synthesize more traffic light instances
    
    # === ADDITIONAL PARAMETERS FOR SMALL OBJECTS ===
    'close_mosaic': 10,    # Disable mosaic last 10 epochs (better convergence)
}

# Dataset path
DATA_YAML = 'LISA_YOLO/lisa.yaml'

# Output directory
OUTPUT_DIR = 'runs/detect/lisa_traffic_lights_optimized'


# ============================================================================
# TRAINING FUNCTION WITH BETTER MONITORING
# ============================================================================

def train_yolo():
    """Train YOLOv8 on LISA dataset with improved settings"""
    print("="*80)
    print("YOLOv8 TRAINING - LISA Traffic Light Detection (OPTIMIZED)")
    print("="*80)
    
    # Verify dataset exists
    if not os.path.exists(DATA_YAML):
        print(f"❌ ERROR: Dataset config not found: {DATA_YAML}")
        print("   Run the converter script first!")
        return None
    
    # Load dataset config
    with open(DATA_YAML, 'r') as f:
        data_config = yaml.safe_load(f)
    
    print(f"\n📊 Dataset Configuration:")
    print(f"   Path: {data_config['path']}")
    print(f"   Classes: {data_config['nc']}")
    print(f"   Names: {', '.join(data_config['names'].values())}")
    
    # Initialize model
    print(f"\n🤖 Initializing {MODEL_SIZE.upper()} model...")
    print(f"   ⚠️  CONSTRAINED to yolov8n due to ESP32-CAM 8MB PSRAM limit")
    print(f"   📋 Compensation strategy:")
    print(f"      • More epochs (200 vs 100)")
    print(f"      • Aggressive augmentation (scale, HSV, copy-paste)")
    print(f"      • Higher classification weight (cls=2.0)")
    print(f"      • Better learning schedule")
    model = YOLO(f'{MODEL_SIZE}.pt')
    
    print(f"\n🚀 Starting training with ESP32-OPTIMIZED settings...")
    print(f"   Key improvements:")
    print(f"   • Model: YOLOv8n (nano) - fits 8MB PSRAM")
    print(f"   • Resolution: 320x320 (ESP32 inference-friendly)")
    print(f"   • More epochs: {CONFIG['epochs']} (compensate for small model)")
    print(f"   • VERY aggressive augmentation (scale=0.9, HSV, copy-paste)")
    print(f"   • High classification weight: cls={CONFIG['cls']} (was 0.5)")
    print(f"   • Higher initial LR: {CONFIG['lr0']} for efficient learning")
    
    # Train model
    results = model.train(
        data=DATA_YAML,
        epochs=CONFIG['epochs'],
        imgsz=CONFIG['imgsz'],
        batch=CONFIG['batch'],
        device=CONFIG['device'],
        workers=CONFIG['workers'],
        patience=CONFIG['patience'],
        save=CONFIG['save'],
        save_period=CONFIG['save_period'],
        cache=CONFIG['cache'],
        resume=CONFIG['resume'],
        amp=CONFIG['amp'],
        pretrained=CONFIG['pretrained'],
        optimizer=CONFIG['optimizer'],
        lr0=CONFIG['lr0'],
        lrf=CONFIG['lrf'],
        momentum=CONFIG['momentum'],
        weight_decay=CONFIG['weight_decay'],
        warmup_epochs=CONFIG['warmup_epochs'],
        warmup_momentum=CONFIG['warmup_momentum'],
        box=CONFIG['box'],
        cls=CONFIG['cls'],
        dfl=CONFIG['dfl'],
        hsv_h=CONFIG['hsv_h'],
        hsv_s=CONFIG['hsv_s'],
        hsv_v=CONFIG['hsv_v'],
        degrees=CONFIG['degrees'],
        translate=CONFIG['translate'],
        scale=CONFIG['scale'],
        shear=CONFIG['shear'],
        perspective=CONFIG['perspective'],
        flipud=CONFIG['flipud'],
        fliplr=CONFIG['fliplr'],
        mosaic=CONFIG['mosaic'],
        mixup=CONFIG['mixup'],
        copy_paste=CONFIG['copy_paste'],
        close_mosaic=CONFIG['close_mosaic'],
        name=OUTPUT_DIR,
        exist_ok=True,
        # Additional parameters for better small object detection
        plots=True,          # Generate training plots
        save_json=True,      # Save results in JSON
        verbose=True,        # Verbose output
    )
    
    print("\n✅ Training complete!")
    print(f"\n📈 Final Results:")
    print(f"   Best mAP@0.5: {results.results_dict['metrics/mAP50(B)']:.3f}")
    print(f"   Best mAP@0.5:0.95: {results.results_dict['metrics/mAP50-95(B)']:.3f}")
    
    return model


# ============================================================================
# VALIDATION WITH LOWER CONFIDENCE THRESHOLD
# ============================================================================

def validate_yolo(model_path='runs/detect/lisa_traffic_lights_optimized/weights/best.pt'):
    """Validate trained model on test set"""
    print("\n" + "="*80)
    print("VALIDATION ON TEST SET")
    print("="*80)
    
    # Load best model
    model = YOLO(model_path)
    
    # Run validation with LOWER confidence threshold
    # Traffic lights are hard to detect, so we need lower threshold
    metrics = model.val(
        data=DATA_YAML,
        split='test',
        imgsz=CONFIG['imgsz'],
        batch=CONFIG['batch'],
        device=CONFIG['device'],
        conf=0.25,  # ← IMPORTANT: Lower confidence threshold
        iou=0.4,    # ← NMS IoU threshold
    )
    
    print(f"\n📊 Test Set Results:")
    print(f"   mAP@0.5: {metrics.box.map50:.3f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.3f}")
    print(f"   Precision: {metrics.box.mp:.3f}")
    print(f"   Recall: {metrics.box.mr:.3f}")
    
    # Per-class results
    print(f"\n📋 Per-Class Results:")
    class_names = ['go', 'goLeft', 'stop', 'stopLeft', 'warning', 'warningLeft']
    for i, name in enumerate(class_names):
        if i < len(metrics.box.ap50):
            print(f"   {name:12s}: mAP@0.5={metrics.box.ap50[i]:.3f}, "
                  f"P={metrics.box.p[i]:.3f}, "
                  f"R={metrics.box.r[i]:.3f}")
    
    return metrics


# ============================================================================
# INFERENCE WITH OPTIMIZED PARAMETERS
# ============================================================================

def test_inference(model_path='runs/detect/lisa_traffic_lights_optimized/weights/best.pt',
                   test_images=['LISA_YOLO/test/images']):
    """Run inference on sample images with optimized detection parameters"""
    print("\n" + "="*80)
    print("INFERENCE DEMO")
    print("="*80)
    
    # Load model
    model = YOLO(model_path)
    
    # Run inference with TRAFFIC-LIGHT-OPTIMIZED parameters
    results = model.predict(
        source=test_images[0],
        imgsz=CONFIG['imgsz'],
        conf=0.2,       # ← LOWER confidence (traffic lights are hard!)
        iou=0.35,       # ← LOWER NMS threshold (avoid suppressing nearby lights)
        max_det=20,     # ← INCREASED (multiple lights in scene)
        save=True,
        save_txt=True,
        show_labels=True,
        show_conf=True,
        line_width=2,   # Thicker boxes for visibility
        agnostic_nms=False,  # Class-aware NMS (don't suppress different colors)
    )
    
    print(f"\n✅ Inference complete! Results saved to runs/detect/predict/")
    return results


# ============================================================================
# EXPORT TO TFLITE WITH PROPER QUANTIZATION
# ============================================================================

def export_to_tflite(model_path='runs/detect/lisa_traffic_lights_optimized/weights/best.pt'):
    """Export trained model to TFLite INT8 with proper calibration"""
    print("\n" + "="*80)
    print("EXPORT TO TFLITE INT8")
    print("="*80)
    
    # Load model
    model = YOLO(model_path)
    
    print(f"\n📦 Exporting to TFLite INT8...")
    print(f"   Input: {model_path}")
    print(f"   Using dataset for quantization calibration...")
    
    # Export to TFLite with INT8 quantization
    export_path = model.export(
        format='tflite',
        imgsz=CONFIG['imgsz'],
        int8=True,
        data=DATA_YAML,  # Crucial: use dataset for proper calibration
        nms=True,        # Include NMS in model (helps with ESP32 deployment)
    )
    
    # Get file size
    file_size_mb = os.path.getsize(export_path) / (1024 * 1024)
    
    print(f"\n✅ Export complete!")
    print(f"   Output: {export_path}")
    print(f"   Size: {file_size_mb:.2f} MB")
    
    # Check if it fits ESP32
    if file_size_mb < 3.0:
        print(f"   ✅ Model fits ESP32-CAM constraints (<3MB)")
    else:
        print(f"   ⚠️  Model might be too large for ESP32-CAM")
        print(f"   Consider using yolov8n instead of yolov8s")
    
    return export_path


# ============================================================================
# DEBUGGING: CHECK WHY PERFORMANCE IS LOW
# ============================================================================

def debug_dataset():
    """Debug dataset to find issues"""
    print("\n" + "="*80)
    print("DATASET DEBUGGING")
    print("="*80)
    
    with open(DATA_YAML, 'r') as f:
        data = yaml.safe_load(f)
    
    # Check class distribution
    print(f"\n🔍 Checking class distribution...")
    
    import glob
    from collections import Counter
    
    for split in ['train', 'val', 'test']:
        label_dir = Path(data['path']) / split / 'labels'
        if not label_dir.exists():
            print(f"   ⚠️  {split} labels not found!")
            continue
        
        label_files = glob.glob(str(label_dir / '*.txt'))
        
        class_counts = Counter()
        bbox_sizes = []
        
        for label_file in label_files:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        w, h = float(parts[3]), float(parts[4])
                        class_counts[class_id] += 1
                        bbox_sizes.append(w * h)
        
        print(f"\n   {split.upper()} split:")
        print(f"   Images: {len(label_files)}")
        print(f"   Total objects: {sum(class_counts.values())}")
        print(f"   Class distribution:")
        for cls_id, count in sorted(class_counts.items()):
            cls_name = data['names'][cls_id]
            print(f"      {cls_name:12s}: {count:5d} ({100*count/sum(class_counts.values()):.1f}%)")
        
        if bbox_sizes:
            import numpy as np
            bbox_sizes = np.array(bbox_sizes)
            print(f"   Bounding box sizes (fraction of image):")
            print(f"      Mean: {bbox_sizes.mean():.4f}")
            print(f"      Median: {np.median(bbox_sizes):.4f}")
            print(f"      Min: {bbox_sizes.min():.4f}, Max: {bbox_sizes.max():.4f}")
            print(f"      < 0.01 (tiny): {100*np.sum(bbox_sizes < 0.01)/len(bbox_sizes):.1f}%")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Complete training and evaluation pipeline"""

    # Step 0: Debug dataset (optional but recommended)
    print("\n🎯 STEP 0: DATASET DEBUGGING")
    debug_dataset()
    
    # Step 1: Train model
    print("\n🎯 STEP 1: TRAINING")
    model = train_yolo()
    
    if model is None:
        print("\n❌ Training failed! Check dataset setup.")
        return
    
    # Step 2: Validate on test set
    print("\n🎯 STEP 2: VALIDATION")
    best_model_path = f'{OUTPUT_DIR}/weights/best.pt'
    metrics = validate_yolo(best_model_path)
    
    # Step 3: Run inference demo
    print("\n🎯 STEP 3: INFERENCE DEMO")
    test_inference(best_model_path)
    
    # Step 4: Export to TFLite
    print("\n🎯 STEP 4: EXPORT TO TFLITE")
    tflite_path = export_to_tflite(best_model_path)
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 COMPLETE WORKFLOW FINISHED!")
    print("="*80)
    print(f"\n📁 Outputs:")
    print(f"   Best model: {best_model_path}")
    print(f"   TFLite INT8: {tflite_path}")
    print(f"\n📊 Expected Improvements (within ESP32 constraints):")
    print(f"   • mAP@0.5: 0.245 → 0.50-0.65 (nano model limit)")
    print(f"   • Recall: 0.29 → 0.60-0.75")
    print(f"   • Model size: <2MB INT8 (fits ESP32-CAM)")
    print(f"   • Better detection of small/distant traffic lights")
    print(f"   ⚠️  Note: Nano model has inherent limitations for tiny objects")
    print("="*80)


if __name__ == "__main__":
    main()