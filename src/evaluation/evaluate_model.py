import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import tensorflow as tf
import os

CSV_PATH = 'data/processed/hybrid_drowsiness_dataset.csv'
MODEL_PATH = 'models/vision_model.h5'

if not os.path.exists(CSV_PATH):
    print(f"Error: Could not find {CSV_PATH}.")
    exit()

print("Loading dataset...")
df = pd.read_csv(CSV_PATH)

# --- RECREATE THE EXACT 50/50 BALANCE USED IN TRAINING ---
print("Balancing the test data to match training...")
df_attentive = df[df['target'] == 0]
df_tired = df[df['target'] == 1]

# Downsample using the EXACT same random_state=42 so the data matches perfectly
df_attentive_downsampled = df_attentive.sample(n=len(df_tired), random_state=42)

# Combine and shuffle
df_balanced = pd.concat([df_attentive_downsampled, df_tired])
df_balanced = shuffle(df_balanced, random_state=42)
# ---------------------------------------------------------

# Drop text columns and isolate features
X = df_balanced.drop(['target', 'img_path'], axis=1).values
y = df_balanced['target'].values

# Split the Data (80% Train, 20% Test) using the same random seed
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Loading model from {MODEL_PATH}...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except OSError:
    print(f"Error: Could not find {MODEL_PATH}.")
    exit()

print("Grading predictions...")
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=1)

# Print Final Report Card
print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, y_pred, target_names=['Attentive (0)', 'Tired (1)']))

# Generate the 2x2 Confusion Matrix Graphic
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Attentive', 'Tired'])

disp.plot(cmap=plt.cm.Blues)
plt.title('Project Aegis: Final Binary Confusion Matrix')
plt.show()