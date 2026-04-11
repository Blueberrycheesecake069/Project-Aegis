import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1. Load Data and Model
CSV_PATH = r'data\processed\new_hybrid_drowsiness_dataset.csv'
MODEL_PATH = r'models\vision_model_v2.h5'

df = pd.read_csv(CSV_PATH)
model = tf.keras.models.load_model(MODEL_PATH)

# 2. Re-create the EXACT same balanced test set used in training
df_attentive = df[df['target'] == 0]
df_tired = df[df['target'] == 1]
min_class_size = min(len(df_attentive), len(df_tired))

df_balanced = pd.concat([
    df_attentive.sample(n=min_class_size, random_state=42),
    df_tired.sample(n=min_class_size, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

cols_to_drop = [col for col in ['target', 'img_path', 'video_name'] if col in df_balanced.columns]
X = df_balanced.drop(cols_to_drop, axis=1).values
y = df_balanced['target'].values

# Grab the 20% test split 
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Generate Predictions
print("Evaluating V2 Model...")
y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)

# 4. Print Professional Metrics
print("\n" + "="*50)
print("PROJECT AEGIS - V2 EVALUATION METRICS")
print("="*50)
print(classification_report(y_test, y_pred_classes, target_names=['Attentive (0)', 'Tired (1)']))

# 5. Display Confusion Matrix (Using standard scikit-learn instead of Seaborn!)
disp = ConfusionMatrixDisplay.from_predictions(
    y_test, 
    y_pred_classes, 
    display_labels=['Attentive', 'Tired'],
    cmap='Blues'
)
disp.ax_.set_title('V2 Model Confusion Matrix')
plt.show()