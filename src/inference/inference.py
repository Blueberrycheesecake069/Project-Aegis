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

# Histories
ear_history = deque(maxlen=WINDOW_SIZE)
mar_history = deque(maxlen=WINDOW_SIZE)
blink_durations = deque(maxlen=10)
yawn_timestamps = deque(maxlen=10)

# State Variables
blink_start_time = None
is_blinking = False
is_yawning = False
missing_face_frames = 0 # Tracks how long you are slumped over

cap = cv2.VideoCapture(0)

# -----------------------------------------------------------
# 3. CALIBRATION PHASE
# -----------------------------------------------------------
print("Starting Calibration... RELAX YOUR FACE AND SIT STRAIGHT.")
calibration_ears = []
calibration_pitches = [] # Track the camera angle

while len(calibration_ears) < CALIBRATION_FRAMES and cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)
    
    if results.face_landmarks:
        h, w, _ = frame.shape
        landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in results.face_landmarks[0]])
        
        # Track EAR
        ear = (calculate_aspect_ratio(landmarks, LEFT_EYE_IDX) + 
               calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)) / 2.0
        calibration_ears.append(ear)
        
        # Track Pitch
        pitch, _, _ = get_head_pose(landmarks, w, h)
        calibration_pitches.append(pitch)
    
    cv2.putText(frame, "CALIBRATING (SIT NORMAL)...", (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.imshow('Project Aegis - Inference', frame)
    cv2.waitKey(1)

baseline_ear = np.mean(calibration_ears) if calibration_ears else 0.3
baseline_pitch = np.mean(calibration_pitches) if calibration_pitches else 0.0
print(f"Calibration Complete. Baseline EAR: {baseline_ear:.3f} | Baseline Pitch: {baseline_pitch:.1f}")

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
        missing_face_frames = 0 # Reset the slump tracker because we see you!
        
        h, w, _ = frame.shape
        landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in results.face_landmarks[0]])
        
        # A. EXTRACT RAW METRICS
        ear = (calculate_aspect_ratio(landmarks, LEFT_EYE_IDX) + 
               calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)) / 2.0
        mar = calculate_aspect_ratio(landmarks, MOUTH_IDX)
        pitch, yaw, roll = get_head_pose(landmarks, w, h)
        
        # THE CAMERA ANGLE FIX: Subtract the physical resting angle
        pitch = pitch - baseline_pitch 
        
        # THE SAFETY CLAMP: Prevent glitchy tracker math from destroying the AI
        pitch = float(np.clip(pitch, -90, 90))
        yaw   = float(np.clip(yaw, -90, 90))
        roll  = float(np.clip(roll, -90, 90))
        
        # B. NORMALIZE
        norm_ear = ear / baseline_ear
        ear_history.append(norm_ear)
        mar_history.append(mar)
        
        # C. EVENT LOGIC
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
                
        if mar > 0.55:
             if not is_yawning:
                is_yawning = True
                yawn_timestamps.append(current_time)
        else:
            is_yawning = False

        # D. PREDICT
        if len(ear_history) == WINDOW_SIZE:
            perclos = np.mean(np.array(ear_history) < 0.65)
            avg_ear_val = np.mean(ear_history)
            ear_std = np.std(ear_history)
            avg_mar_val = np.mean(mar_history)
            blink_rate = len([e for e in ear_history if e < 0.65]) / (WINDOW_SIZE / 20.0) 
            avg_blink_dur = np.mean(blink_durations) if blink_durations else 0.0
            final_blink_dur = max(avg_blink_dur, current_active_blink_dur)
            recent_yawns = len([t for t in yawn_timestamps if t > current_time - 3.0])

            features = np.array([[perclos, avg_ear_val, ear_std, avg_mar_val, recent_yawns, blink_rate, final_blink_dur, pitch, yaw, roll]])
            
            # THE X-RAY
            print(f"NormEAR: {norm_ear:.2f} | PERCLOS: {perclos:.2f} | BlinkDur: {final_blink_dur:.2f} | Pitch: {pitch:.1f}")
            
            prediction_probs = model.predict(features, verbose=0)[0]
            class_idx = np.argmax(prediction_probs)
            confidence = prediction_probs[class_idx]
            
            # --- THE EXTREME HYBRID OVERRIDE ---
            # The AI handles normal drowsiness. This ONLY triggers for literal sleep.
            if final_blink_dur > 2.5 or perclos > 0.85:
                class_idx = 1
                confidence = 0.99
                state_text = "!!! MICRO-SLEEP DETECTED !!!"
                color = (0, 0, 255)
            # -----------------------------------
            elif class_idx == 0:
                state_text = f"ATTENTIVE ({int(confidence*100)}%)"
                color = (0, 255, 0)
            elif class_idx == 1:
                state_text = f"TIRED ({int(confidence*100)}%)"
                color = (0, 0, 255)
                if confidence > 0.8: state_text = "!!! WAKE UP !!!"

            if pitch < -15: state_text += " [HEAD NOD]"

    else:
        # THE MISSING FACE ALARM: Trigger if you slump completely out of view
        missing_face_frames += 1
        if missing_face_frames > 15: # About 0.5 seconds of missing face
            state_text = "!!! FACE LOST / HEAD DOWN !!!"
            color = (0, 0, 255)

    # DRAW OVERLAY
    cv2.putText(frame, state_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.imshow('Project Aegis - Inference', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()