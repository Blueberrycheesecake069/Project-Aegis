import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.feature_utils import (
    calculate_aspect_ratio, get_head_pose,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX
)

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
# Paths
YAWDD_DIR = r'C:\Users\achintya\driver_drowsiness_system\data\external\YawDD' 
# *** CHANGE THIS TO YOUR ACTUAL UTA CSV NAME ***
UTA_CSV_PATH = r'C:\Users\achintya\driver_drowsiness_system\data\processed\hybrid_drowsiness_dataset.csv'
OUTPUT_CSV = r'C:\Users\achintya\driver_drowsiness_system\data\processed\new_hybrid_drowsiness_dataset.csv'

# MediaPipe Setup
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

WINDOW_SIZE = 60         
Generic_Baseline_EAR = 0.30 

# The EXACT column order you requested
COLS = [
    'perclos', 'norm_avg_ear', 'ear_std', 'avg_mar', 'yawn_count', 
    'blink_rate', 'avg_blink_dur', 'pitch', 'yaw', 'roll', 
    'img_path', 'target'
]

def process_yawdd_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return []

    ear_history = deque(maxlen=WINDOW_SIZE)
    mar_history = deque(maxlen=WINDOW_SIZE)
    blink_durations = deque(maxlen=20)
    yawn_timestamps = deque(maxlen=20)
    
    blink_start_time = None
    is_blinking = False
    is_yawning = False
    video_rows = []
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    print(f"Extracting: {os.path.basename(video_path)} | Binary Label: {label}")

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame_idx += 1
        current_time = frame_idx / fps
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)
        
        if results.face_landmarks:
            h, w, _ = frame.shape
            landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in results.face_landmarks[0]])
            
            raw_ear = (calculate_aspect_ratio(landmarks, LEFT_EYE_IDX) + calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)) / 2.0
            mar = calculate_aspect_ratio(landmarks, MOUTH_IDX)
            pitch, yaw, roll = get_head_pose(landmarks, w, h)
            
            norm_ear = raw_ear / Generic_Baseline_EAR
            ear_history.append(norm_ear)
            mar_history.append(mar)
            
            if norm_ear < 0.50:
                if not is_blinking:
                    is_blinking = True
                    blink_start_time = current_time
            else:
                if is_blinking:
                    is_blinking = False
                    blink_durations.append(current_time - blink_start_time)

            if mar > 0.55:
                if not is_yawning:
                    is_yawning = True
                    yawn_timestamps.append(current_time)
            else:
                is_yawning = False
            
            if len(ear_history) == WINDOW_SIZE:
                perclos = np.mean(np.array(ear_history) < 0.50)
                avg_ear_val = np.mean(ear_history)
                ear_std = np.std(ear_history)
                avg_mar_val = np.mean(mar_history)
                blink_rate = len([e for e in ear_history if e < 0.50]) / (WINDOW_SIZE / fps)
                avg_blink_dur = np.mean(blink_durations) if blink_durations else 0.0
                recent_yawns = len([t for t in yawn_timestamps if t > current_time - 3.0])
                
                # Order strictly matches the COLS list
                video_rows.append([
                    perclos, avg_ear_val, ear_std, avg_mar_val, recent_yawns, 
                    blink_rate, avg_blink_dur, pitch, yaw, roll, 
                    os.path.basename(video_path), label
                ])
                
    cap.release()
    return video_rows

# -----------------------------------------------------------------------------
# 2. EXTRACT YAWDD DATA (Upgraded Radar Search)
# -----------------------------------------------------------------------------
print("--- PHASE 1: EXTRACTING YAWDD VIDEOS ---")
yawdd_data = []

# os.walk automatically searches every folder and subfolder inside YawDD
for root, dirs, files in os.walk(YAWDD_DIR):
    for file in files:
        if file.lower().endswith(('.mp4', '.avi', '.mov')):
            video_path = os.path.join(root, file)
            
            # Binary Smart Label: Yawn = 1 (Tired), Everything else = 0 (Attentive)
            label = 1 if 'yawn' in file.lower() else 0
            yawdd_data.extend(process_yawdd_video(video_path, label))

df_yawdd = pd.DataFrame(yawdd_data, columns=COLS)
print(f"YawDD Extraction Complete: {len(df_yawdd)} rows generated.")

# -----------------------------------------------------------------------------
# 3. LOAD UTA DATA & CREATE HYBRID CSV
# -----------------------------------------------------------------------------
print("\n--- PHASE 2: MERGING WITH UTA DATA ---")
try:
    df_uta = pd.read_csv(UTA_CSV_PATH)
    
    # Ensure UTA columns match exactly just in case
    df_uta = df_uta[COLS] 
    print(f"Loaded UTA Data: {len(df_uta)} rows.")
    
    # FUSE THEM: YawDD first, then UTA
    df_hybrid = pd.concat([df_yawdd, df_uta], ignore_index=True)
    
    df_hybrid.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSUCCESS! Hybrid dataset created perfectly.")
    print(f"Total Rows: {len(df_hybrid)}")
    print(f"Saved to: {OUTPUT_CSV}")

except Exception as e:
    print(f"\n[!] Error loading or merging the UTA CSV. Did you update the UTA_CSV_PATH?")
    print(f"Details: {e}")