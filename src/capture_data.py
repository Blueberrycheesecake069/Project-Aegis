import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.feature_utils import calculate_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX

# Ensure directories exist for the hybrid approach
os.makedirs('data/raw/eye_images', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

# 1. SETUP MODERN FACELANDMARKER (2026 Tasks API)
# Ensure 'face_landmarker.task' is inside your 'models/' folder
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE, # Optimized for camera frames
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Sliding Window Setup (Pipeline 3 Stage 7)
WINDOW_SIZE = 600 
ear_history = deque(maxlen=WINDOW_SIZE)
mar_history = deque(maxlen=WINDOW_SIZE)

cap = cv2.VideoCapture(0)
data_rows = []
label = 1  # MANUALLY CHANGE TO 1 FOR DROWSY DATA
click_count = 0

print(f"REC: Recording for Label {label}. Press 's' to save data, 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    # MediaPipe Tasks requires RGB and a specific Image object
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Perform Detection
    results = detector.detect(mp_image)
    
    if results.face_landmarks:
        h, w, _ = frame.shape
        # Extract the first face's landmarks
        face_landmarks = results.face_landmarks[0]
        landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks])
        
        # 2. COMPUTATION (Pipeline 3 Stages 4-5)
        left_ear = calculate_aspect_ratio(landmarks, LEFT_EYE_IDX)
        right_ear = calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)
        ear = (left_ear + right_ear) / 2.0
        mar = calculate_aspect_ratio(landmarks, MOUTH_IDX)
        
        ear_history.append(ear)
        mar_history.append(mar)
        
        # 3. REFINED PARAMETERS (Pipeline 3 Stage 7)
        if len(ear_history) == WINDOW_SIZE:
            perclos = np.mean(np.array(ear_history) < 0.21)
            avg_ear = np.mean(ear_history)
            max_mar = np.max(mar_history)
            blink_rate = len([e for e in ear_history if e < 0.15]) / 30.0 
            current_features = [perclos, avg_ear, max_mar, blink_rate]
            
            # HYBRID FEATURE: Eye Crop for CNN Branch
            eye_pts = landmarks[LEFT_EYE_IDX + RIGHT_EYE_IDX]
            ex, ey, ew, eh = cv2.boundingRect(eye_pts)
            # Crop with a small margin
            eye_crop = frame[max(0, ey-30):min(h, ey+eh+30), max(0, ex-30):min(w, ex+ew+30)]
            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                img_filename = f"eye_{label}_{click_count}.jpg"
                # Add image path to the 4 numerical features
                data_rows.append(current_features + [img_filename, label])
                
                # Save the image patch for the Hybrid CNN branch
                if eye_crop.size > 0:
                    eye_resized = cv2.resize(eye_crop, (64, 64)) 
                    cv2.imwrite(f'data/raw/eye_images/{img_filename}', eye_resized)
                    click_count += 1
                    print(f"Saved {click_count}/50 for Label {label}")

    cv2.imshow('Project Aegis - Hybrid Collector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# 4. SAVE TO CSV
df = pd.DataFrame(data_rows, columns=['perclos', 'avg_ear', 'max_mar', 'blink_rate', 'img_path', 'target'])
csv_path = 'data/processed/hybrid_drowsiness_dataset.csv'
df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))

cap.release()
cv2.destroyAllWindows()