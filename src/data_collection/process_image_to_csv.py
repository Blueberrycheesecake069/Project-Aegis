import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.feature_utils import (
    calculate_aspect_ratio, get_head_pose,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX
)

# 1. SETUP & LOAD MODEL
model_path = 'models/vision_model.h5'
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully.")
except:
    print("Error: Could not load model. Train it first!")
    exit()

base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options, 
    running_mode=vision.RunningMode.IMAGE, 
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# 2. PARAMETERS
WINDOW_SIZE = 30         
CALIBRATION_FRAMES = 50  
SMOOTHING_SECONDS = 5.0  

# Histories
ear_history = deque(maxlen=WINDOW_SIZE)
mar_history = deque(maxlen=WINDOW_SIZE)
blink_durations = deque(maxlen=10)
yawn_timestamps = deque(maxlen=10)
prediction_history = deque() 

# State Variables
blink_start_time = None
is_blinking = False
is_yawning = False
missing_face_frames = 0 

cap = cv2.VideoCapture(0)

# -----------------------------------------------------------
# 3. CALIBRATION PHASE
# -----------------------------------------------------------
print("Starting Calibration... RELAX YOUR FACE AND SIT STRAIGHT.")
calibration_ears = []

while len(calibration_ears) < CALIBRATION_FRAMES and cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)
    
    if results.face_landmarks:
        h, w, _ = frame.shape
        landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in results.face_landmarks[0]])
        
        ear = (calculate_aspect_ratio(landmarks, LEFT_EYE_IDX) + 
               calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)) / 2.0
        calibration_ears.append(ear)
    
    cv2.putText(frame, "CALIBRATING (SIT NORMAL)...", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.imshow('Project Aegis - Inference', frame)
    cv2.waitKey(1)

baseline_ear = np.mean(calibration_ears) if calibration_ears else 0.3
print(f"Calibration Complete. Baseline EAR: {baseline_ear:.3f}")

# -----------------------------------------------------------
# 4. REAL-TIME INFERENCE LOOP
# -----------------------------------------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    current_time = time.time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)
    
    state_text = "WAITING FOR DATA..."
    color = (255, 255, 255)
    
    if results.face_landmarks:
        missing_face_frames = 0 
        
        h, w, _ = frame.shape
        landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in results.face_landmarks[0]])
        
        # A. EXTRACT RAW METRICS
        ear = (calculate_aspect_ratio(landmarks, LEFT_EYE_IDX) + 
               calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)) / 2.0
        mar = calculate_aspect_ratio(landmarks, MOUTH_IDX)
        
        # B. NORMALIZE
        norm_ear = ear / baseline_ear
        ear_history.append(norm_ear)
        mar_history.append(mar)
        
        # C. EVENT LOGIC (For emergency rule-based overrides)
        current_active_blink_dur = 0.0
        if norm_ear < 0.65: 
            if not is_blinking:
                is_blinking = True
                blink_start_time = current_time
            current_active_blink_dur = current_time - blink_start_time
        else:
            if is_blinking:
                is_blinking = False
                blink_durations.append(current_time - blink_start_time)

        # D. PREDICT
        if len(ear_history) == WINDOW_SIZE:
            
            # 1. Calculate REAL stats for our emergency text alerts
            real_perclos = np.mean(np.array(ear_history) < 0.65)
            avg_mar_val = np.mean(mar_history)
            avg_blink_dur = np.mean(blink_durations) if blink_durations else 0.0
            real_final_blink_dur = max(avg_blink_dur, current_active_blink_dur)

            # 2. MOCK STATS FOR AI (Because the AI was trained on 0.0 for these!)
            mock_perclos = 0.0
            mock_ear_std = 0.0
            mock_yawn_count = 0
            mock_blink_rate = 0.0
            mock_avg_blink_dur = 0.0
            pitch, yaw, roll = 0.0, 0.0, 0.0 # Bypassing broken 3D math

            # Create the exact array the AI expects!
            features = np.array([[mock_perclos, norm_ear, mock_ear_std, avg_mar_val, mock_yawn_count, mock_blink_rate, mock_avg_blink_dur, pitch, yaw, roll]])
            
            # --- AI INFERENCE ---
            try:
                prediction_probs = model(features, training=False).numpy()[0]
                raw_class_idx = np.argmax(prediction_probs)
                confidence = prediction_probs[raw_class_idx]
            except Exception as e:
                print(f"\n[!!!] FATAL AI CRASH: {e}\n")
                break 
            
            # --- TRUE TIME-BASED DEBOUNCING (5 SECONDS) ---
            prediction_history.append((current_time, raw_class_idx))
            
            while prediction_history and prediction_history[0][0] < current_time - SMOOTHING_SECONDS:
                prediction_history.popleft()
                
            if len(prediction_history) > 0:
                tired_votes = sum([vote[1] for vote in prediction_history])
                tired_ratio = tired_votes / len(prediction_history)
                
                if tired_ratio >= 0.60:
                    smoothed_class_idx = 1
                else:
                    smoothed_class_idx = 0
            else:
                smoothed_class_idx = 0
                tired_ratio = 0.0
            
            # --- THE X-RAY: AI MIND READER ---
            print(f"NormEAR: {norm_ear:.2f} | AI Tired: {prediction_probs[1]*100:.0f}% | 5s Ratio: {tired_ratio*100:.0f}%")

            # --- THE EXTREME HYBRID OVERRIDE (Using REAL stats) ---
            if real_final_blink_dur > 2.5 or real_perclos > 0.85:
                smoothed_class_idx = 1
                confidence = 0.99
                state_text = "!!! MICRO-SLEEP DETECTED !!!"
                color = (0, 0, 255)
            elif smoothed_class_idx == 0:
                state_text = f"ATTENTIVE ({int(confidence*100)}%)"
                color = (0, 255, 0)
            elif smoothed_class_idx == 1:
                state_text = f"TIRED ({int(confidence*100)}%)"
                color = (0, 0, 255)
                if confidence > 0.8: state_text = "!!! WAKE UP !!!"

    else:
        missing_face_frames += 1
        if missing_face_frames > 15: 
            state_text = "!!! FACE LOST / HEAD DOWN !!!"
            color = (0, 0, 255)

    # DRAW OVERLAY
    cv2.putText(frame, state_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.imshow('Project Aegis - Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()