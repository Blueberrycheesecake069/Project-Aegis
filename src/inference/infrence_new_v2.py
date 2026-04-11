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
    calculate_aspect_ratio,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX
)

# -----------------------------------------------------------
# 1. NEW STABLE 3D HEAD POSE MATH
# -----------------------------------------------------------
def get_head_pose_stable(landmarks, img_w, img_h):
    face_3d = np.array([
        (0.0, 0.0, 0.0),            # Nose tip (1)
        (0.0, -330.0, -65.0),       # Chin (152)
        (-225.0, 170.0, -135.0),    # Left eye corner (33)
        (225.0, 170.0, -135.0),     # Right eye corner (263)
        (-150.0, -150.0, -125.0),   # Left mouth corner (61)
        (150.0, -150.0, -125.0)     # Right mouth corner (291)
    ], dtype=np.float64)

    face_2d = np.array([
        landmarks[1], landmarks[152], landmarks[33], 
        landmarks[263], landmarks[61], landmarks[291]
    ], dtype=np.float64)

    focal_length = 1.0 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2],
                           [0, focal_length, img_w / 2],
                           [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    if not success: return 0.0, 0.0, 0.0
    
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    # FIX: Removed the extra [0] to stop the float subscriptable crash
    return angles[0], angles[1], angles[2] 

# -----------------------------------------------------------
# 2. SETUP & LOAD V2 MODEL
# -----------------------------------------------------------
model_path = 'models/vision_model_v2.h5'
try:
    model = tf.keras.models.load_model(model_path)
    print("Project Aegis V2 Model loaded successfully.")
except Exception as e:
    print(f"Error: Could not load model. Check path! {e}")
    exit()

base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(
    base_options=base_options, 
    running_mode=vision.RunningMode.IMAGE, 
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# -----------------------------------------------------------
# 3. STABLE BINARY PARAMETERS
# -----------------------------------------------------------
FEATURE_WINDOW = 60          
CALIBRATION_FRAMES = 50
SMOOTHING_SECONDS = 5.0      
BINARY_THRESHOLD = 0.60      

# Histories
ear_history = deque(maxlen=FEATURE_WINDOW)
mar_history = deque(maxlen=FEATURE_WINDOW)
blink_durations = deque(maxlen=20)
yawn_timestamps = deque(maxlen=20)
prediction_history = deque() 

# State Variables
blink_start_time = None
is_blinking = False
is_yawning = False
missing_face_frames = 0 

cap = cv2.VideoCapture(0)

# -----------------------------------------------------------
# 4. CALIBRATION PHASE
# -----------------------------------------------------------
print("\nStarting Calibration... RELAX YOUR FACE AND SIT STRAIGHT.")
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

baseline_ear = np.mean(calibration_ears) if calibration_ears else 0.30
print(f"Calibration Complete. Your natural baseline EAR is: {baseline_ear:.3f}\n")

# -----------------------------------------------------------
# 5. REAL-TIME INFERENCE LOOP
# -----------------------------------------------------------
while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    current_time = time.time()
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)
    
    state_text = "GATHERING DATA..."
    color = (255, 255, 255)
    
    if results.face_landmarks:
        missing_face_frames = 0 
        h, w, _ = frame.shape
        landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in results.face_landmarks[0]])
        
        # A. EXTRACT RAW METRICS
        raw_ear = (calculate_aspect_ratio(landmarks, LEFT_EYE_IDX) + 
                   calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)) / 2.0
        mar = calculate_aspect_ratio(landmarks, MOUTH_IDX)
        
        # USE THE NEW STABLE FUNCTION HERE:
        pitch, yaw, roll = get_head_pose_stable(landmarks, w, h)
        pitch = float(np.clip(pitch, -90, 90))
        
        # B. NORMALIZE
        norm_ear = raw_ear / baseline_ear
        ear_history.append(norm_ear)
        mar_history.append(mar)
        
        # C. EVENT LOGIC 
        current_active_blink_dur = 0.0
        if norm_ear < 0.50: 
            if not is_blinking:
                is_blinking = True
                blink_start_time = current_time
            current_active_blink_dur = current_time - blink_start_time
        else:
            if is_blinking:
                is_blinking = False
                blink_durations.append(current_time - blink_start_time)

        # UPDATED YAWN THRESHOLD (0.80)
        if mar > 0.80:
             if not is_yawning:
                is_yawning = True
                yawn_timestamps.append(current_time)
        else:
            is_yawning = False

        # D. PREDICT & DEBOUNCE
        if len(ear_history) == FEATURE_WINDOW:
            
            perclos = np.mean(np.array(ear_history) < 0.50)
            avg_ear_val = np.mean(ear_history)
            ear_std = np.std(ear_history)
            avg_mar_val = np.mean(mar_history)
            blink_rate = len([e for e in ear_history if e < 0.50]) / (FEATURE_WINDOW / fps)
            avg_blink_dur = np.mean(blink_durations) if blink_durations else 0.0
            recent_yawns = len([t for t in yawn_timestamps if t > current_time - 3.0]) 
            
            features = np.array([[perclos, avg_ear_val, ear_std, avg_mar_val, recent_yawns, 
                                  blink_rate, avg_blink_dur, pitch, yaw, roll]])
            
            prediction_probs = model(features, training=False).numpy()[0]
            ai_confidence = prediction_probs[1]
            
            is_tired_frame = 1 if ai_confidence > 0.55 else 0
            
            # --- THE 5-SECOND TEMPORAL WINDOW ---
            prediction_history.append((current_time, is_tired_frame))
            
            while prediction_history and prediction_history[0][0] < current_time - SMOOTHING_SECONDS:
                prediction_history.popleft()
                
            tired_votes = sum([vote[1] for vote in prediction_history])
            tired_ratio = tired_votes / len(prediction_history) if len(prediction_history) > 0 else 0.0
            
            real_final_blink_dur = max(avg_blink_dur, current_active_blink_dur)
            
            if real_final_blink_dur > 2.5 or perclos > 0.85:
                state_text = "!!! MICRO-SLEEP DETECTED !!!"
                color = (0, 0, 255)
            elif tired_ratio >= BINARY_THRESHOLD:
                state_text = f"DROWSY ALARM ({int(tired_ratio*100)}%)"
                color = (0, 0, 255)
            else:
                state_text = f"ATTENTIVE ({100 - int(tired_ratio*100)}%)"
                color = (0, 255, 0)

            if pitch < -15: state_text += " [HEAD NOD]"
            
            print(f"FEAT: PERC:{perclos:.2f} EAR:{avg_ear_val:.2f} MAR:{avg_mar_val:.2f} YAWN:{recent_yawns} PITCH:{pitch:.1f} | AI RAW: {ai_confidence*100:.0f}% | 5s RATIO: {tired_ratio*100:.0f}%")

    else:
        missing_face_frames += 1
        if missing_face_frames > 15: 
            state_text = "!!! FACE LOST / HEAD DOWN !!!"
            color = (0, 0, 255)

    cv2.putText(frame, state_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.imshow('Project Aegis - Inference', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

print("\nShutting down Project Aegis...")
cap.release()
cv2.destroyAllWindows()