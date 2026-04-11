import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# ==========================================
# PROJECT AEGIS - BAYES ERROR ESTIMATOR
# Based on ICLR 2023: "Is the Performance of My Deep Network Too Good to be True?"
# ==========================================

# 1. Load your trained model
model_path = 'models/vision_model.h5'
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 2. Load your validation/test CSV (the one your video script generated)
# IMPORTANT: Point this to your actual test data CSV!
csv_path = 'data/processed/hybrid_drowsiness_dataset.csv' 
try:
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded: {len(df)} rows.")
except Exception as e:
    print(f"Error loading CSV. Make sure the path is correct! {e}")
    exit()

# 3. Replicate the exact train/test split used during training — evaluate on held-out 20% only
df_attentive = df[df['target'] == 0]
df_tired = df[df['target'] == 1]
min_class_size = min(len(df_attentive), len(df_tired))
df_attentive_down = df_attentive.sample(n=min_class_size, random_state=42)
df_tired_down = df_tired.sample(n=min_class_size, random_state=42)
df_balanced = shuffle(pd.concat([df_attentive_down, df_tired_down]), random_state=42)

cols_to_drop = [col for col in ['target', 'img_path', 'video_name'] if col in df_balanced.columns]
X_all = df_balanced.drop(cols_to_drop, axis=1).values
y_all = df_balanced['target'].values

_, X_test, _, _ = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)
print(f"Held-out test set size: {len(X_test)} rows (20% of balanced dataset).")

# 4. Get the AI's "Soft Labels" (Probabilities)
print("Calculating prediction probabilities...")
predictions = model.predict(X_test, verbose=0)

# If your model outputs 2 columns (Awake vs Tired), grab the probability of class 1
if predictions.shape[1] > 1:
    c_i = predictions[:, 1]
else:
    c_i = predictions[:, 0]

# 5. THE ICLR 2023 BAYES ERROR FORMULA
# Formula: mean of the minimum between the probability and (1 - probability)
bayes_errors = np.minimum(c_i, 1 - c_i)
estimated_bayes_error = np.mean(bayes_errors)

# 6. The Grand Reveal
print(f"\n" + "="*50)
print(f"PROJECT AEGIS - DATASET CEILING ANALYSIS")
print(f"="*50)
print(f"Total samples evaluated: {len(c_i)}")
print(f"Estimated Bayes Error: {estimated_bayes_error * 100:.2f}%")
print(f"Absolute Maximum Theoretical Accuracy: {(1 - estimated_bayes_error) * 100:.2f}%")
print(f"="*50 + "\n")