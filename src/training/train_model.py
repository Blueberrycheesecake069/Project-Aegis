import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import os

CSV_PATH = 'data/processed/hybrid_drowsiness_dataset.csv'
MODEL_SAVE_PATH = 'models/vision_model.h5'

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

print("Balancing the dataset...")
df_attentive = df[df['target'] == 0]
df_tired = df[df['target'] == 1]

# Downsample Attentive to perfectly match Tired
df_attentive_downsampled = df_attentive.sample(n=len(df_tired), random_state=42)

df_balanced = pd.concat([df_attentive_downsampled, df_tired])
df_balanced = shuffle(df_balanced, random_state=42)

# Drop text columns and isolate features
X = df_balanced.drop(['target', 'img_path'], axis=1).values
y = df_balanced['target'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- THE MAGIC FIX: BAKE A SCALER INTO THE MODEL ---
print("Calculating data scales (this takes a few seconds)...")
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(X_train) # This learns the min/max of your specific features
# ---------------------------------------------------

print("Building the AI...")
model = tf.keras.Sequential([
    normalizer, # Automatically squashes the numbers before they hit the neurons!
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2), 
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Starting training...")
os.makedirs('models', exist_ok=True)

history = model.fit(
    X_train, y_train,
    epochs=20, 
    batch_size=256,
    validation_data=(X_test, y_test)
)

model.save(MODEL_SAVE_PATH)
print(f"Model saved successfully to {MODEL_SAVE_PATH}!")