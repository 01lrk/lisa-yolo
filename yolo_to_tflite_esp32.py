"""
YOLOv8 to TFLite Converter for ESP32 (Windows-Compatible)
Fixed version with proper dependency handling
"""

from ultralytics import YOLO
import tensorflow as tf
import numpy as np
import os
import subprocess
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

YOLO_MODEL_PATH = 'runs/detect/runs/detect/lisa_traffic_lights/weights/best.pt'
OUTPUT_DIR = 'esp32_models'
INPUT_SIZE = 320
CALIBRATION_DATA = 'LISA_YOLO/lisa.yaml'

# ============================================================================
# DEPENDENCY CHECKER
# ============================================================================

def install_dependencies():
    """Install required dependencies"""
    dependencies = {
        'onnx': 'onnx',
        'onnx2tf': 'onnx2tf',
        'sng4onnx': 'sng4onnx>=1.0.1',
    }
    
    print("\n[Checking dependencies...]")
    
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError:
            print(f"  ✗ {module} - Installing...")
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                             check=True, capture_output=True)
                print(f"    ✓ Installed {package}")
            except Exception as e:
                print(f"    ✗ Failed to install {package}: {e}")
                return False
    
    return True


# ============================================================================
# METHOD 1: ONNX2TF (Works on Windows without ai-edge-litert)
# ============================================================================

def convert_via_onnx2tf(model_path, output_dir, input_size=320):
    """
    Use onnx2tf library - Works on Windows!
    This avoids the ai-edge-litert dependency issue
    """
    print("\n" + "="*80)
    print("CONVERSION METHOD: ONNX → TensorFlow (via onnx2tf)")
    print("="*80)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 1: Export to ONNX
    print("\n[1/4] Exporting YOLO to ONNX...")
    model = YOLO(model_path)
    
    try:
        # Export with higher opset for better compatibility
        onnx_path = model.export(format='onnx', imgsz=input_size, opset=13, simplify=True)
        print(f"✅ ONNX export successful: {onnx_path}")
    except Exception as e:
        print(f"❌ ONNX export failed: {e}")
        return None
    
    # Step 2: Convert ONNX to TensorFlow SavedModel using onnx2tf
    print("\n[2/4] Converting ONNX to TensorFlow SavedModel...")
    
    try:
        import onnx2tf
        
        saved_model_dir = os.path.join(output_dir, 'saved_model')
        
        # Convert ONNX to TF
        onnx2tf.convert(
            input_onnx_file_path=onnx_path,
            output_folder_path=saved_model_dir,
            copy_onnx_input_output_names_to_tflite=True,
            non_verbose=False
        )
        
        print(f"✅ SavedModel created: {saved_model_dir}")
        
    except ImportError:
        print("Installing onnx2tf...")
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-U', 'onnx2tf'], check=True)
            import onnx2tf
            
            saved_model_dir = os.path.join(output_dir, 'saved_model')
            onnx2tf.convert(
                input_onnx_file_path=onnx_path,
                output_folder_path=saved_model_dir,
                copy_onnx_input_output_names_to_tflite=True,
                non_verbose=False
            )
            print(f"✅ SavedModel created: {saved_model_dir}")
        except Exception as e:
            print(f"❌ onnx2tf installation/conversion failed: {e}")
            return convert_manual_tflite(onnx_path, output_dir, input_size)
    except Exception as e:
        print(f"❌ Conversion failed: {e}")
        return convert_manual_tflite(onnx_path, output_dir, input_size)
    
    # Step 3: Convert SavedModel to TFLite
    print("\n[3/4] Converting to TFLite FP32...")
    
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        tflite_fp32 = converter.convert()
        
        fp32_path = os.path.join(output_dir, 'yolo_fp32.tflite')
        with open(fp32_path, 'wb') as f:
            f.write(tflite_fp32)
        
        size_mb = os.path.getsize(fp32_path) / (1024*1024)
        print(f"✅ FP32 model saved: {fp32_path} ({size_mb:.2f} MB)")
    except Exception as e:
        print(f"❌ FP32 conversion failed: {e}")
        fp32_path = None
    
    # Step 4: Quantize to INT8
    print("\n[4/4] Converting to TFLite INT8...")
    
    int8_path = quantize_to_int8(saved_model_dir, output_dir, input_size)
    
    if int8_path:
        print(f"\n{'='*80}")
        print("✅ CONVERSION COMPLETE!")
        print(f"{'='*80}")
        
        if fp32_path:
            fp32_size = os.path.getsize(fp32_path) / (1024*1024)
            print(f"FP32: {fp32_path} ({fp32_size:.2f} MB)")
        
        int8_size = os.path.getsize(int8_path) / (1024*1024)
        print(f"INT8: {int8_path} ({int8_size:.2f} MB) ← Use this for ESP32")
        
        check_esp32_compatibility(int8_path)
        return int8_path
    elif fp32_path:
        return fp32_path
    
    return None


# ============================================================================
# METHOD 2: Direct TFLite from ONNX (Fallback)
# ============================================================================

def convert_manual_tflite(onnx_path, output_dir, input_size):
    """
    Direct ONNX to TFLite using tf2onnx reverse (if onnx2tf fails)
    """
    print("\n[Fallback] Trying direct ONNX to TFLite...")
    
    try:
        import onnx
        
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Try using onnx-tensorflow (older but sometimes works)
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', 
                          'tensorflow-addons', 'onnx-tensorflow'], check=True)
            
            from onnx_tf.backend import prepare
            
            tf_rep = prepare(onnx_model)
            saved_model_dir = os.path.join(output_dir, 'saved_model_fallback')
            tf_rep.export_graph(saved_model_dir)
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            
            tflite_model = converter.convert()
            tflite_path = os.path.join(output_dir, 'yolo_fallback.tflite')
            
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✅ Fallback conversion successful: {tflite_path}")
            return tflite_path
            
        except Exception as e:
            print(f"❌ onnx-tensorflow failed: {e}")
            return None
            
    except Exception as e:
        print(f"❌ Fallback conversion failed: {e}")
        return None


# ============================================================================
# QUANTIZATION
# ============================================================================

def quantize_to_int8(saved_model_path, output_dir, input_size):
    """Quantize SavedModel to INT8 TFLite"""
    import glob
    
    # Representative dataset generator
    def representative_dataset_gen():
        import cv2
        
        calib_paths = [
            'LISA_YOLO/train/images/*.jpg',
            'LISA_YOLO/val/images/*.jpg',
            'LISA_YOLO/train/images/*.png',
        ]
        
        images = []
        for pattern in calib_paths:
            found = glob.glob(pattern)
            images.extend(found)
            if len(images) >= 100:
                break
        
        if len(images) == 0:
            print("  ⚠️ No calibration images found, using random data")
            for _ in range(100):
                yield [np.random.rand(1, input_size, input_size, 3).astype(np.float32)]
        else:
            print(f"  📊 Using {min(len(images), 100)} images for calibration")
            count = 0
            for img_path in images[:100]:
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (input_size, input_size))
                    img = img.astype(np.float32) / 255.0
                    img = np.expand_dims(img, axis=0)
                    yield [img]
                    count += 1
                    if count % 20 == 0:
                        print(f"    Processed {count}/100 images...")
                except Exception:
                    continue
    
    # Try dynamic range quantization first (faster, still good compression)
    print("  Attempting dynamic range quantization...")
    
    try:
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        
        tflite_quant = converter.convert()
        quant_path = os.path.join(output_dir, 'yolo_int8_dynamic.tflite')
        
        with open(quant_path, 'wb') as f:
            f.write(tflite_quant)
        
        size_mb = os.path.getsize(quant_path) / (1024*1024)
        print(f"  ✅ Dynamic quantization successful ({size_mb:.2f} MB)")
        
        # Try full integer quantization for better compression
        print("  Attempting full integer quantization...")
        
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset_gen
        
        # Allow hybrid (some ops in float if needed)
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS
        ]
        
        tflite_int8 = converter.convert()
        int8_path = os.path.join(output_dir, 'yolo_int8_full.tflite')
        
        with open(int8_path, 'wb') as f:
            f.write(tflite_int8)
        
        int8_size_mb = os.path.getsize(int8_path) / (1024*1024)
        print(f"  ✅ Full integer quantization successful ({int8_size_mb:.2f} MB)")
        
        # Return smaller model
        if int8_size_mb < size_mb:
            print(f"  Using full INT8 model (smaller)")
            return int8_path
        else:
            print(f"  Using dynamic quantization model (smaller)")
            return quant_path
        
    except Exception as e:
        print(f"  ❌ Full quantization failed: {e}")
        
        # Return dynamic quantization if it worked
        if os.path.exists(quant_path):
            return quant_path
        
        return None


def check_esp32_compatibility(tflite_path):
    """Check if model fits ESP32 PSRAM"""
    size_bytes = os.path.getsize(tflite_path)
    size_mb = size_bytes / (1024 * 1024)
    
    PSRAM_TOTAL = 8.0
    RUNTIME_MULTIPLIER = 1.5
    FIRMWARE_OVERHEAD = 1.5
    
    runtime_needed = size_mb * RUNTIME_MULTIPLIER + FIRMWARE_OVERHEAD
    
    print(f"\n{'='*80}")
    print("ESP32 COMPATIBILITY CHECK")
    print(f"{'='*80}")
    print(f"Model size: {size_mb:.2f} MB")
    print(f"Runtime estimate: {runtime_needed:.2f} MB")
    print(f"PSRAM available: {PSRAM_TOTAL:.2f} MB")
    
    if runtime_needed < PSRAM_TOTAL:
        margin = PSRAM_TOTAL - runtime_needed
        print(f"\n✅ COMPATIBLE! {margin:.2f} MB free memory")
        print(f"   Safety margin: {(margin/PSRAM_TOTAL)*100:.1f}%")
    else:
        deficit = runtime_needed - PSRAM_TOTAL
        print(f"\n❌ TOO LARGE by {deficit:.2f} MB")
        print("\n💡 Solutions:")
        print("  1. Reduce INPUT_SIZE to 256 or 224")
        print("  2. Use YOLOv8n (nano) instead of larger variants")
        print("  3. Train with fewer classes if possible")
        print("  4. Consider MobileNetV2-SSD (much smaller)")


def test_inference(tflite_path):
    """Quick inference test"""
    print(f"\n{'='*80}")
    print("INFERENCE TEST")
    print(f"{'='*80}")
    
    try:
        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"Model: {os.path.basename(tflite_path)}")
        print(f"Input: {input_details[0]['shape']} ({input_details[0]['dtype']})")
        print(f"Output: {output_details[0]['shape']} ({output_details[0]['dtype']})")
        
        # Test inference
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        
        if input_dtype == np.uint8:
            test_input = np.random.randint(0, 255, input_shape, dtype=np.uint8)
        else:
            test_input = np.random.rand(*input_shape).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], test_input)
        
        import time
        start = time.time()
        interpreter.invoke()
        inference_time = (time.time() - start) * 1000
        
        output = interpreter.get_tensor(output_details[0]['index'])
        
        print(f"\n✅ Inference successful!")
        print(f"   PC time: {inference_time:.1f}ms")
        print(f"   ESP32 estimate: ~{inference_time * 8:.0f}ms ({1000/(inference_time*8):.1f} FPS)")
        
        return True
        
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
        return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("="*80)
    print("YOLO → TFLITE CONVERTER (Windows-Compatible)")
    print("="*80)
    print("Using onnx2tf method (works without ai-edge-litert)")
    
    if not os.path.exists(YOLO_MODEL_PATH):
        print(f"\n❌ Model not found: {YOLO_MODEL_PATH}")
        print("\nMake sure you have trained a YOLO model first.")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Convert model
    tflite_model = convert_via_onnx2tf(YOLO_MODEL_PATH, OUTPUT_DIR, INPUT_SIZE)
    
    if tflite_model:
        # Test the model
        test_inference(tflite_model)
        
        print(f"\n{'='*80}")
        print("🎉 SUCCESS!")
        print(f"{'='*80}")
        print(f"\n📦 Your model: {tflite_model}")
        print(f"\n📋 Next steps:")
        print(f"  1. Copy {os.path.basename(tflite_model)} to your ESP32 project")
        print(f"  2. Use ESP-IDF with TensorFlow Lite Micro")
        print(f"  3. Example: https://github.com/espressif/esp-tflite-micro")
        
    else:
        print(f"\n{'='*80}")
        print("❌ CONVERSION FAILED")
        print(f"{'='*80}")
        print("\n🔧 ALTERNATIVE OPTIONS:")
        print("\n1. Use Google Colab (Linux, has all dependencies):")
        print("   - Upload your .pt file to Google Drive")
        print("   - Run: model.export(format='tflite', int8=True)")
        print("\n2. Use WSL2 (Windows Subsystem for Linux):")
        print("   - Install: wsl --install")
        print("   - Run conversion in Ubuntu")
        print("\n3. Online converter:")
        print("   - Upload ONNX to https://convertmodel.com")
        print("   - Convert ONNX → TFLite")


if __name__ == "__main__":
    main()