import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

# 1. LOAD DATA
personal_df = pd.read_csv('data/processed/hybrid_drowsiness_dataset.csv')
X_ext = np.load('data/external/BlinksTest_30_Fold1.npy')
y_ext_raw = np.load('data/external/LabelsTest_30_Fold1.npy')

# 2. PREPROCESS EXTERNAL
X_ext_avg = np.mean(X_ext, axis=1)
y_ext = np.where(y_ext_raw > 0, 1, 0)

# 3. THE "HONESTY" SPLIT (80% for training, 20% for pure evaluation)
X_train_ext, X_test_ext, y_train_ext, y_test_ext = train_test_split(
    X_ext_avg, y_ext, test_size=0.2, random_state=42
)

# 4. CREATE TRAINING SET (Personal + 80% External)
ext_train_df = pd.DataFrame(X_train_ext, columns=['perclos', 'avg_ear', 'max_mar', 'blink_rate'])
ext_train_df['target'] = y_train_ext
master_train_df = pd.concat([personal_df, ext_train_df], ignore_index=True).sample(frac=1)

# 5. CREATE EVALUATION SET (Only the 20% Unseen External)
master_test_df = pd.DataFrame(X_test_ext, columns=['perclos', 'avg_ear', 'max_mar', 'blink_rate'])
master_test_df['target'] = y_test_ext

# 6. SAVE BOTH
os.makedirs('data/processed', exist_ok=True)
master_train_df.to_csv('data/processed/master_train.csv', index=False)
master_test_df.to_csv('data/processed/master_test.csv', index=False)

print(f"Split Complete! Training samples: {len(master_train_df)} | Test samples: {len(master_test_df)}")