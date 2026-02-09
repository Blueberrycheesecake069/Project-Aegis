import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.feature_utils import calculate_aspect_ratio, LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX

# 1. SETUP
model = tf.keras.models.load_model('models/vision_model.h5')
base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
options = vision.FaceLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.IMAGE, num_faces=1)
detector = vision.FaceLandmarker.create_from_options(options)

# 2. REACTIVE PARAMETERS (No 30-second wait)
FAST_WINDOW = 40  # Only ~2 seconds of "memory" at 20 FPS
ear_history = deque(maxlen=FAST_WINDOW)
mar_history = deque(maxlen=FAST_WINDOW)

# Counters for the 2-second rule
closure_counter = 0 
FPS = 20 
DROWSY_LIMIT = FPS * 2.0  # 40 consecutive frames of closed eyes

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results = detector.detect(mp_image)
    
    # DEFAULT STATE: AWAKE
    current_state = "AWAKE"
    ui_color = (0, 255, 0) # Green

    if results.face_landmarks:
        face_landmarks = results.face_landmarks[0]
        landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in face_landmarks])
        
        ear = (calculate_aspect_ratio(landmarks, LEFT_EYE_IDX) + 
               calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)) / 2.0
        mar = calculate_aspect_ratio(landmarks, MOUTH_IDX)
        
        ear_history.append(ear)
        mar_history.append(mar)

        # --- LOGIC A: THE 2-SECOND RULE (Immediate Reactive State) ---
        if ear < 0.18:
            closure_counter += 1
        else:
            closure_counter = 0 # Reset on blink

        if closure_counter >= DROWSY_LIMIT:
            current_state = "DROWSY (MICROSLEEP!)"
            ui_color = (0, 0, 255) # Red
        
        # --- LOGIC B: SHORT-TERM FATIGUE (Neural Network) ---
        # This only triggers if the eyes are "droopy" over the last 2 seconds
        elif len(ear_history) == FAST_WINDOW:
            perclos = np.mean(np.array(ear_history) < 0.21)
            avg_ear = np.mean(ear_history)
            max_mar = np.max(mar_history)
            blink_rate = len([e for e in ear_history if e < 0.15]) # Rapid blinks in 2s
            
            features = np.array([[perclos, avg_ear, max_mar, blink_rate]])
            prediction = model.predict(features, verbose=0)[0][0]
            
            if prediction > 0.8: # High confidence only
                current_state = "DROWSY (FATIGUE)"
                ui_color = (0, 165, 255) # Orange

    # Display only the current active state
    cv2.putText(frame, f"STATE: {current_state}", (30, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, ui_color, 3)
    
    cv2.imshow('Project Aegis - Reactive Monitor', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()