import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os

# 1. LOAD MODEL
model_path = 'models/vision_model.h5'
if not os.path.exists(model_path):
    print("Error: vision_model.h5 not found in models/ directory.")
    exit()

model = tf.keras.models.load_model(model_path)

# 2. LOAD EXTERNAL UTA-RLDD DATA
# Ensure these files are in your data/external/ folder
try:
    X_external = np.load('data/external/BlinksTest_30_Fold1.npy')
    y_external_raw = np.load('data/external/LabelsTest_30_Fold1.npy')
except FileNotFoundError:
    print("Error: External .npy files not found. Check data/external/ path.")
    exit()

# 3. FIX: RESHAPE DATA (The "Incompatible Shape" Fix)
# Your model expects (None, 4). The dataset provides (None, 30, 4).
# We average the 30 frames to get a single feature set per sample.
X_external_fixed = np.mean(X_external, axis=1) 

# 4. MAP LABELS TO BINARY (0 or 1)
# UTA-RLDD labels 5 and 10 mean drowsiness; we map them to 1.
y_external = np.where(y_external_raw > 0, 1, 0)

# 5. GENERATE PREDICTIONS
print(f"Evaluating model on {len(X_external_fixed)} fresh samples from UTA-RLDD...")
y_probs = model.predict(X_external_fixed, verbose=1)
y_pred = (y_probs > 0.5).astype(int)

# 6. PERFORMANCE REPORT
print("\n" + "="*40)
print("PROJECT AEGIS: EXTERNAL VALIDATION REPORT")
print("="*40)

acc = accuracy_score(y_external, y_pred)
print(f"Global Benchmark Accuracy: {acc*100:.2f}%\n")

print("Detailed Classification Report:")
print(classification_report(y_external, y_pred, target_names=['Awake', 'Drowsy']))

print("Confusion Matrix:")
cm = confusion_matrix(y_external, y_pred)
print(cm)