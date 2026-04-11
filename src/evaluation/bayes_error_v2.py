import pandas as pd
import numpy as np
import tensorflow as tf

# 1. Paths to your newly fused data and V2 brain
CSV_PATH = r'data\processed\new_hybrid_drowsiness_dataset.csv'
MODEL_PATH = r'models\vision_model_v2.h5'

print("Loading Hybrid Dataset and V2 Model...")
try:
    df = pd.read_csv(CSV_PATH)
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"Error loading files: {e}")
    exit()

# 2. Clean the data (Drop text columns, keep only the 10 math features)
cols_to_drop = [col for col in ['target', 'img_path', 'video_name'] if col in df.columns]
X_eval = df.drop(cols_to_drop, axis=1).values

print("Scanning the full V2 dataset for uncertainty (this might take a few seconds)...")

# 3. Generate predictions for the entire dataset
predictions = model.predict(X_eval, verbose=1)

# Grab the probability of Class 0 (Attentive) for every single row
c_i = predictions[:, 0]  

# 4. ICLR 2023 Bayes Error Formula
# The formula calculates the minimum between the probability of class 0 and class 1 for every sample, 
# then takes the average. This represents the inherent ambiguity in the dataset.
bayes_errors = np.minimum(c_i, 1 - c_i)
estimated_bayes_error = np.mean(bayes_errors)

# 5. Print the Diagnostics
print("\n" + "="*55)
print(" PROJECT AEGIS V2 - BAYES ERROR DIAGNOSTICS")
print("="*55)
print(f"Total frames evaluated:       {len(c_i):,}")
print(f"Estimated Bayes Error:        {estimated_bayes_error * 100:.2f}%")
print("-" * 55)
print(f"Theoretical Maximum Accuracy: {(1 - estimated_bayes_error) * 100:.2f}%")
print("="*55 + "\n")