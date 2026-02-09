import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

# 1. LOAD AND AUGMENT DATA
csv_path = 'data/processed/hybrid_drowsiness_dataset.csv'
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found. Please run capture_data.py first.")
    exit()

df = pd.read_csv(csv_path)

# Synthetic Extreme Values for Emergency Robustness
# PERCLOS, AVG_EAR, MAX_MAR, BLINK_RATE, img_path, target
extreme_data = pd.DataFrame([
    [1.0, 0.05, 3.0, 0.0, 'synthetic', 1],  # Total blackout/eyes closed
    [0.95, 0.08, 0.5, 25.0, 'synthetic', 1], # Heavy drooping with panic blinks
    [1.0, 0.02, 1.2, 0.0, 'synthetic', 1],  # Dead sleep
    [0.0, 0.35, 0.3, 5.0, 'synthetic', 0],   # Perfect alertness
], columns=df.columns)

# Combine real data with synthetic extremes
df_augmented = pd.concat([df, extreme_data], ignore_index=True)

X = df_augmented[['perclos', 'avg_ear', 'max_mar', 'blink_rate']].values
y = df_augmented['target'].values

# 2. DATA SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. BUILD TINY NEURAL NETWORK (PIPELINE 3 STAGE 8)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(4,)),           # 4 features input
    tf.keras.layers.Dense(8, activation='relu'),  # Hidden Layer: 8 neurons
    tf.keras.layers.Dense(4, activation='relu'),  # Stability Layer
    tf.keras.layers.Dense(1, activation='sigmoid') # VisionScore Output (0-1)
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. TRAIN MODEL
print(f"Training on {len(X_train)} samples...")
model.fit(X_train, y_train, epochs=100, batch_size=8, validation_data=(X_test, y_test), verbose=1)

# 5. SAVE MODEL (FIXED DIRECTORY LOGIC)
if not os.path.exists('models'):
    os.makedirs('models')

model.save('models/vision_model.h5')
print("\nSuccess: Model saved to 'models/vision_model.h5'")