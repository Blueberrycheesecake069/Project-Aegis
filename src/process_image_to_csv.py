import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.feature_utils import calculate_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX

# Setup Modern MediaPipe Tasks
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

image_dir = 'data/raw/eye_images/'
all_processed_rows = [] # This must stay outside the loop to collect everything

print(f"Starting batch process for images in {image_dir}...")

for filename in os.listdir(image_dir):
    if filename.endswith(".jpg"):
        img_path = os.path.join(image_dir, filename)
        frame = cv2.imread(img_path)
        
        if frame is None: continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Perform Detection
        results = detector.detect(mp_image)
        
        if results.face_landmarks:
            h, w, _ = frame.shape
            face_landmarks = results.face_landmarks[0]
            landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks])
            
            # Extract EAR/MAR for this specific frame
            left_ear = calculate_aspect_ratio(landmarks, LEFT_EYE_IDX)
            right_ear = calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)
            ear = (left_ear + right_ear) / 2.0
            mar = calculate_aspect_ratio(landmarks, MOUTH_IDX)
            
            # Determine Label from filename
            label = 1 if "eye_1_" in filename else 0
            
            # For static images, we use the single-frame EAR as a proxy for window stats
            # [PERCLOS, AVG_EAR, MAX_MAR, BLINK_RATE, PATH, TARGET]
            row = [ear, ear, mar, 0, filename, label]
            all_processed_rows.append(row)
            print(f"Processed: {filename} | Label: {label}")

# Final Step: Save the full list of 100 rows
if all_processed_rows:
    df = pd.DataFrame(all_processed_rows, columns=['perclos', 'avg_ear', 'max_mar', 'blink_rate', 'img_path', 'target'])
    df.to_csv('data/processed/hybrid_drowsiness_dataset.csv', index=False)
    print(f"Success! Saved {len(all_processed_rows)} rows to CSV.")
else:
    print("Error: No landmarks were detected in any images.")