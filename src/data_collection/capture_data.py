import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import time
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
# Ensure you have updated feature_utils.py first to include get_head_pose
from utils.feature_utils import (
    calculate_aspect_ratio, get_head_pose, 
    LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX
)

# -----------------------------------------------------------------------------
# 1. SETUP
# -----------------------------------------------------------------------------
os.makedirs('data/raw/eye_images', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)

base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Parameters
WINDOW_SIZE = 600      # ~30 seconds of history at 20 FPS
CALIBRATION_FRAMES = 100 # ~5 seconds for baseline

# Histories for sliding window calculation
ear_history = deque(maxlen=WINDOW_SIZE)
mar_history = deque(maxlen=WINDOW_SIZE)
blink_durations = deque(maxlen=20)  # Store last 20 blink durations
yawn_timestamps = deque(maxlen=20)  # Store timestamps of recent yawns

# State Variables for Temporal Events
blink_start_time = None
is_blinking = False
is_yawning = False

cap = cv2.VideoCapture(0)
data_rows = []
label = 1  # <--- MANUALLY CHANGE: 0 for Awake, 1 for Drowsy
click_count = 0

# -----------------------------------------------------------------------------
# 2. CALIBRATION PHASE
# -----------------------------------------------------------------------------
print("=================================================")
print("  PROJECT AEGIS: CALIBRATION PHASE")
print("  Please look at the camera naturally (AWAKE mode).")
print("=================================================")

calibration_ears = []
while len(calibration_ears) < CALIBRATION_FRAMES and cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)
    
    if results.face_landmarks:
        face_landmarks = results.face_landmarks[0]
        h, w, _ = frame.shape
        landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks])
        
        left_ear = calculate_aspect_ratio(landmarks, LEFT_EYE_IDX)
        right_ear = calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)
        calibration_ears.append((left_ear + right_ear) / 2.0)
    
    # Visual Feedback
    cv2.putText(frame, f"Calibrating: {len(calibration_ears)}/{CALIBRATION_FRAMES}", (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    cv2.imshow('Project Aegis - Hybrid Collector', frame)
    cv2.waitKey(1)

# Calculate Personal Baseline
baseline_ear = np.mean(calibration_ears) if calibration_ears else 0.3
print(f"DONE. Personal Baseline EAR set to: {baseline_ear:.3f}")
print(f"REC: Recording for Label {label}. Press 's' to save, 'q' to quit.")

# -----------------------------------------------------------------------------
# 3. DATA COLLECTION LOOP
# -----------------------------------------------------------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    current_time = time.time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)
    
    if results.face_landmarks:
        h, w, _ = frame.shape
        face_landmarks = results.face_landmarks[0]
        landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks])
        
        # --- A. FEATURE EXTRACTION ---
        # 1. Geometric Ratios
        left_ear = calculate_aspect_ratio(landmarks, LEFT_EYE_IDX)
        right_ear = calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)
        raw_ear = (left_ear + right_ear) / 2.0
        mar = calculate_aspect_ratio(landmarks, MOUTH_IDX)
        
        # 2. Normalize EAR
        norm_ear = raw_ear / baseline_ear
        
        # 3. Head Pose (3D)
        pitch, yaw, roll = get_head_pose(landmarks, w, h)
        
        # Update Histories
        ear_history.append(norm_ear)
        mar_history.append(mar)
        
        # --- B. TEMPORAL LOGIC (Blinks & Yawns) ---
        # Blink Duration Logic
        if norm_ear < 0.50: # Threshold relative to baseline (50% closure)
            if not is_blinking:
                is_blinking = True
                blink_start_time = current_time
        else:
            if is_blinking:
                is_blinking = False
                duration = current_time - blink_start_time
                blink_durations.append(duration)

        # Yawn Counting Logic
        if mar > 0.55: # Threshold for Yawn
            if not is_yawning:
                is_yawning = True
                yawn_timestamps.append(current_time)
        else:
            is_yawning = False

        # --- C. COMPUTE FINAL FEATURE VECTOR ---
        if len(ear_history) == WINDOW_SIZE:
            # 1. Ocular Features
            perclos = np.mean(np.array(ear_history) < 0.50) # % time < 50% open
            avg_ear_val = np.mean(ear_history)
            ear_std = np.std(ear_history)
            
            # Blink Frequency & Duration
            blink_rate = len([e for e in ear_history if e < 0.50]) / (WINDOW_SIZE / 20.0) # Approx per sec
            avg_blink_dur = np.mean(blink_durations) if blink_durations else 0.0
            
            # 2. Mouth Features
            avg_mar_val = np.mean(mar_history)
            # Count yawns in the last 30 seconds (Window duration)
            window_duration_est = WINDOW_SIZE / 20.0
            recent_yawns = len([t for t in yawn_timestamps if t > current_time - window_duration_est])
            
            # 3. Head Pose Features
            # (Pitch, Yaw, Roll already calculated)

            # THE 10-FEATURE VECTOR
            current_features = [
                perclos, avg_ear_val, ear_std,          # Eye Stats
                avg_mar_val, recent_yawns,              # Mouth Stats
                blink_rate, avg_blink_dur,              # Blink Dynamics
                pitch, yaw, roll                        # Head Pose
            ]
            
            # --- D. SAVE DATA ---
            # Crop Eye for Hybrid CNN
            eye_pts = landmarks[LEFT_EYE_IDX + RIGHT_EYE_IDX]
            ex, ey, ew, eh = cv2.boundingRect(eye_pts)
            eye_crop = frame[max(0, ey-30):min(h, ey+eh+30), max(0, ex-30):min(w, ex+ew+30)]
            
            if cv2.waitKey(1) & 0xFF == ord('s'):
                img_filename = f"eye_{label}_{click_count}.jpg"
                
                # Append Path and Target
                data_rows.append(current_features + [img_filename, label])
                
                if eye_crop.size > 0:
                    eye_resized = cv2.resize(eye_crop, (64, 64)) 
                    cv2.imwrite(f'data/raw/eye_images/{img_filename}', eye_resized)
                    click_count += 1
                    print(f"Saved {click_count} | Pitch:{int(pitch)} Roll:{int(roll)}")

    # Visual UI
    cv2.putText(frame, f"Label: {label} | Count: {click_count}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow('Project Aegis - Hybrid Collector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

# 4. SAVE TO CSV
columns = [
    'perclos', 'norm_avg_ear', 'ear_std', 
    'avg_mar', 'yawn_count', 
    'blink_rate', 'avg_blink_dur', 
    'pitch', 'yaw', 'roll', 
    'img_path', 'target'
]

df = pd.DataFrame(data_rows, columns=columns)
csv_path = 'data/processed/hybrid_drowsiness_dataset.csv'
# Append if exists, write header only if not
df.to_csv(csv_path, mode='a', index=False, header=not os.path.exists(csv_path))

print(f"Session Complete. Saved {len(data_rows)} rows.")
cap.release()
cv2.destroyAllWindows()