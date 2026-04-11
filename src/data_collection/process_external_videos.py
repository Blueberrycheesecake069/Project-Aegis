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
from utils.feature_utils import (
    calculate_aspect_ratio, get_head_pose,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX
)

# -----------------------------------------------------------------------------
# 1. SETUP & CONFIGURATION
# -----------------------------------------------------------------------------
# Path to your downloaded video folders
DATASET_PATH = 'data/external/videos/' 

# Output file
OUTPUT_CSV = 'data/processed/hybrid_drowsiness_dataset.csv'

# MediaPipe Setup
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE, # Use IMAGE mode for video files to be safe
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# Parameters (Must match training logic)
WINDOW_SIZE = 60         # 3 seconds history @ 20FPS
Generic_Baseline_EAR = 0.30 # Standard "Awake" EAR for normalization

def process_video_file(video_path, label):
    """
    Watches a single video file and extracts features row-by-row.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening {video_path}")
        return []

    # Temporal Histories for this specific video
    ear_history = deque(maxlen=WINDOW_SIZE)
    mar_history = deque(maxlen=WINDOW_SIZE)
    blink_durations = deque(maxlen=20)
    yawn_timestamps = deque(maxlen=20)
    
    # State tracking
    blink_start_time = None
    is_blinking = False
    is_yawning = False
    
    video_rows = []
    frame_idx = 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0: fps = 30.0 # Fallback
    
    print(f"Processing: {os.path.basename(video_path)} | Label: {label}")

    while True:
        success, frame = cap.read()
        if not success: break
        
        frame_idx += 1
        
        # Current "Time" in the video (seconds)
        current_time = frame_idx / fps
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        results = detector.detect(mp_image)
        
        if results.face_landmarks:
            h, w, _ = frame.shape
            landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in results.face_landmarks[0]])
            
            # --- FEATURE EXTRACTION ---
            left_ear = calculate_aspect_ratio(landmarks, LEFT_EYE_IDX)
            right_ear = calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)
            raw_ear = (left_ear + right_ear) / 2.0
            mar = calculate_aspect_ratio(landmarks, MOUTH_IDX)
            pitch, yaw, roll = get_head_pose(landmarks, w, h)
            
            # Normalize
            norm_ear = raw_ear / Generic_Baseline_EAR
            
            # Update History
            ear_history.append(norm_ear)
            mar_history.append(mar)
            
            # --- TEMPORAL LOGIC ---
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
            
            # --- SAVE DATA POINT ---
            if len(ear_history) == WINDOW_SIZE:
                perclos = np.mean(np.array(ear_history) < 0.50)
                avg_ear_val = np.mean(ear_history)
                ear_std = np.std(ear_history)
                avg_mar_val = np.mean(mar_history)
                
                # Blink Rate (per second)
                blink_rate = len([e for e in ear_history if e < 0.50]) / (WINDOW_SIZE / fps)
                avg_blink_dur = np.mean(blink_durations) if blink_durations else 0.0
                recent_yawns = len([t for t in yawn_timestamps if t > current_time - 3.0])
                
                row = [
                    perclos, avg_ear_val, ear_std, 
                    avg_mar_val, recent_yawns, 
                    blink_rate, avg_blink_dur, 
                    pitch, yaw, roll, 
                    os.path.basename(video_path), label
                ]
                video_rows.append(row)
                
    cap.release()
    return video_rows

# -----------------------------------------------------------------------------
# 2. MAIN LOOP - UPDATED FOR 3 CLASSES
# -----------------------------------------------------------------------------
all_data = []

# Class 0: Attentive (Green)
folder_0 = os.path.join(DATASET_PATH, 'attentive')
if os.path.exists(folder_0):
    for vid in os.listdir(folder_0):
        if vid.endswith(('.mp4', '.avi', '.mov')):
            rows = process_video_file(os.path.join(folder_0, vid), 0)
            all_data.extend(rows)
else:
    print(f"Warning: Folder not found: {folder_0}")

# Class 1: Low Vigilance (Orange)
folder_1 = os.path.join(DATASET_PATH, 'low_vigilance')
if os.path.exists(folder_1):
    for vid in os.listdir(folder_1):
        if vid.endswith(('.mp4', '.avi', '.mov')):
            rows = process_video_file(os.path.join(folder_1, vid), 1)
            all_data.extend(rows)
else:
    print(f"Warning: Folder not found: {folder_1}")

# Class 2: Drowsy (Red)
folder_2 = os.path.join(DATASET_PATH, 'drowsy')
if os.path.exists(folder_2):
    for vid in os.listdir(folder_2):
        if vid.endswith(('.mp4', '.avi', '.mov')):
            rows = process_video_file(os.path.join(folder_2, vid), 2)
            all_data.extend(rows)
else:
    print(f"Warning: Folder not found: {folder_2}")

# -----------------------------------------------------------------------------
# 3. SAVE TO CSV
# -----------------------------------------------------------------------------
if all_data:
    cols = [
        'perclos', 'norm_avg_ear', 'ear_std', 
        'avg_mar', 'yawn_count', 
        'blink_rate', 'avg_blink_dur', 
        'pitch', 'yaw', 'roll', 
        'img_path', 'target'
    ]
    df = pd.DataFrame(all_data, columns=cols)
    
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"SUCCESS: Processed {len(all_data)} frames of professional data.")
    print(f"Saved to {OUTPUT_CSV}")
else:
    print("No data processed. Check your paths!")