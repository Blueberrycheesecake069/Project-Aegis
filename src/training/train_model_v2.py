import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os

# -----------------------------------------------------------------------------
# 1. SETUP & LOADING
# -----------------------------------------------------------------------------
CSV_PATH = 'data/processed/hybrid_drowsiness_dataset.csv'
MODEL_SAVE_PATH = 'models/vision_model_v2.h5'  # Saving as V2 to protect your old prototype!

print("Loading Master Hybrid Dataset...")
try:
    df = pd.read_csv(CSV_PATH)
    print(f"Total raw rows loaded: {len(df)}")
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# -----------------------------------------------------------------------------
# 2. THE BALANCER (Handling the massive UTA + YawDD imbalance)
# -----------------------------------------------------------------------------
print("\nBalancing the dataset...")
df_attentive = df[df['target'] == 0]
df_tired = df[df['target'] == 1]

print(f"Original Distribution -> Attentive (0): {len(df_attentive)} | Tired (1): {len(df_tired)}")

# Downsample the massive majority class to perfectly match the minority class
# This naturally restricts the massive UTA dataset while using 100% of the YawDD minority data!
min_class_size = min(len(df_attentive), len(df_tired))

df_attentive_downsampled = df_attentive.sample(n=min_class_size, random_state=42)
df_tired_downsampled = df_tired.sample(n=min_class_size, random_state=42)

df_balanced = pd.concat([df_attentive_downsampled, df_tired_downsampled])
df_balanced = shuffle(df_balanced, random_state=42).reset_index(drop=True)

print(f"Balanced Dataset Size: {len(df_balanced)} rows ({min_class_size} per class)")

# -----------------------------------------------------------------------------
# 3. PREPARE DATA FOR AI
# -----------------------------------------------------------------------------
# Drop text columns dynamically (in case some rows have 'video_name' instead of 'img_path')
cols_to_drop = [col for col in ['target', 'img_path', 'video_name'] if col in df_balanced.columns]

X = df_balanced.drop(cols_to_drop, axis=1).values
y = df_balanced['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# -----------------------------------------------------------------------------
# 4. BAKE THE SCALER INTO THE MODEL
# -----------------------------------------------------------------------------
print("\nCalculating data scales (Normalizer)...")
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train) 

# -----------------------------------------------------------------------------
# 5. UPGRADED V2 ARCHITECTURE (Increased Capacity + BatchNorm)
# -----------------------------------------------------------------------------
print("Building the V2 AI Brain...")
model = tf.keras.Sequential([
    normalizer, 
    
    # Layer 1: Increased from 64 to 128 to handle combined UTA/YawDD complexity
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(), # Prevents "Extreme Ends" predictions
    tf.keras.layers.Dropout(0.3), 
    
    # Layer 2: Increased from 32 to 64
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    
    # Layer 3: Increased from 16 to 32
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    
    # Output Layer: Binary Classification
    tf.keras.layers.Dense(2, activation='softmax')
])

# -----------------------------------------------------------------------------
# 6. ADVANCED TRAINING LOGIC
# -----------------------------------------------------------------------------
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Add Callbacks to prevent overfitting and stop training when perfect
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=7, restore_best_weights=True, verbose=1
)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, verbose=1
)

print("\nStarting Training for Project Aegis V2...")
os.makedirs('models', exist_ok=True)

history = model.fit(
    X_train, y_train,
    epochs=40,            # Increased epochs since EarlyStopping will catch it
    batch_size=256,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]
)

model.save(MODEL_SAVE_PATH)
print(f"\n==================================================")
print(f"SUCCESS! V2 Model saved to: {MODEL_SAVE_PATH}")
print(f"==================================================")