"""
Run this ONCE on your PC to convert vision_model.h5 -> vision_model.tflite
before copying the project to the Raspberry Pi.

Usage (from project root):
    python convert_to_tflite.py
"""

import os
import tensorflow as tf

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
H5_PATH      = os.path.join(PROJECT_ROOT, 'models', 'vision_model.h5')
TFLITE_PATH  = os.path.join(PROJECT_ROOT, 'models', 'vision_model.tflite')

print(f"Loading model from: {H5_PATH}")
model = tf.keras.models.load_model(H5_PATH)
model.summary()

print("\nConverting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Optional: float16 quantisation — halves model size, negligible accuracy loss
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# converter.target_spec.supported_types = [tf.float16]

tflite_model = converter.convert()

with open(TFLITE_PATH, 'wb') as f:
    f.write(tflite_model)

size_kb = os.path.getsize(TFLITE_PATH) / 1024
print(f"\nDone. Saved to: {TFLITE_PATH}")
print(f"Model size: {size_kb:.1f} KB")
print("\nCopy the entire project folder to your Pi and run:")
print("  python src/inference/inference_debounced_pi.py")
