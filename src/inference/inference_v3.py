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

# =============================================================================
# KALMAN FILTER (1D)
# Separates real physiological signal change from MediaPipe landmark jitter.
# =============================================================================
class KalmanFilter1D:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2):
        self.Q = process_variance       # expected true signal change per frame
        self.R = measurement_variance   # measurement noise
        self.P = 1.0                    # initial estimation uncertainty
        self.x = None                   # state estimate

    def update(self, z):
        if self.x is None:
            self.x = z
            return self.x
        self.P = self.P + self.Q
        K = self.P / (self.P + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * self.P
        return self.x

# =============================================================================
# 1. SETUP & LOAD MODEL
# =============================================================================
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

# =============================================================================
# 2. PARAMETERS
# =============================================================================
WINDOW_SIZE          = 30
CALIBRATION_FRAMES   = 50
SMOOTHING_SECONDS    = 5.0
WARMUP_SECONDS       = 5.0

# Base thresholds — these are progressively adapted during the session
BASE_EAR_CLOSED      = 0.75   # normalized EAR below this = eye closed
BASE_VOTE_THRESHOLD  = 0.50   # tired fraction in voting window to trigger alert

# Session-adaptive sensitivity parameters
# Every 60 minutes: EAR threshold rises by ADAPT_EAR_RATE, vote threshold drops by ADAPT_VOTE_RATE
ADAPT_EAR_RATE       = 0.08   # max +0.08 on EAR_CLOSED over 60 min (catches lighter closure)
ADAPT_VOTE_RATE      = 0.12   # max -0.12 on vote threshold over 60 min (more trigger-happy)
EAR_CLOSED_MAX       = 0.85   # hard cap on EAR_CLOSED threshold
VOTE_THRESHOLD_MIN   = 0.35   # hard floor on vote threshold

# Periodic baseline recalibration
RECALIBRATE_INTERVAL = 300    # seconds between baseline recalibration attempts (5 min)
RECALIBRATE_MIN_SAMPLES = 60  # minimum confident-attentive frames needed to recalibrate
RECALIBRATE_MAX_DRIFT = 0.20  # reject recalibration if new baseline drifts >20% from original

# Alert history weighting
ALERT_HISTORY_WINDOW = 600    # seconds to look back for alert history (10 min)
ALERT_BOOST_PER_ALERT = 0.025 # vote threshold reduction per recent alert
ALERT_BOOST_MAX      = 0.10   # cap on total alert-history boost

# EAR slope onset detection
SLOPE_ONSET_THRESHOLD = -0.003  # per-frame EAR drop rate to flag as onset (negative = falling)
SLOPE_BOOST          = 0.08    # vote threshold reduction when onset detected

# =============================================================================
# 3. HISTORIES & STATE
# =============================================================================
ear_history        = deque(maxlen=WINDOW_SIZE)
mar_history        = deque(maxlen=WINDOW_SIZE)
blink_durations    = deque(maxlen=10)
yawn_timestamps    = deque(maxlen=10)
prediction_history = deque()
alert_timestamps   = deque()              # timestamps of every TIRED alert fired
confident_att_ears = deque(maxlen=300)    # EAR values from high-confidence ATTENTIVE frames

blink_start_time   = None
is_blinking        = False
is_yawning         = False
missing_face_frames = 0
window_filled_time  = None
last_recalibrate_time = None
session_start_time  = None   # set after calibration completes

ear_kf   = KalmanFilter1D(process_variance=1e-4, measurement_variance=1e-2)
pitch_kf = KalmanFilter1D(process_variance=0.5,  measurement_variance=2.0)

# =============================================================================
# 4. CALIBRATION PHASE
# =============================================================================
cap = cv2.VideoCapture(0)

print("Starting Calibration... RELAX YOUR FACE AND SIT STRAIGHT.")
calibration_ears    = []
calibration_pitches = []

while len(calibration_ears) < CALIBRATION_FRAMES and cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results   = detector.detect(mp_image)

    if results.face_landmarks:
        h, w, _ = frame.shape
        landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in results.face_landmarks[0]])

        ear = (calculate_aspect_ratio(landmarks, LEFT_EYE_IDX) +
               calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)) / 2.0
        calibration_ears.append(ear)

        pitch, _, _ = get_head_pose(landmarks, w, h)
        calibration_pitches.append(pitch)

    cv2.putText(frame, "CALIBRATING (SIT NORMAL)...", (50, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    cv2.imshow('Project Aegis V3', frame)
    cv2.waitKey(1)

baseline_ear         = np.mean(calibration_ears)   if calibration_ears    else 0.3
baseline_pitch       = np.mean(calibration_pitches) if calibration_pitches else 0.0
original_baseline_ear = baseline_ear  # kept for drift sanity-check during recalibration
session_start_time   = time.time()
last_recalibrate_time = session_start_time

print(f"Calibration Complete. Baseline EAR: {baseline_ear:.3f} | Baseline Pitch: {baseline_pitch:.1f}")

# =============================================================================
# 5. REAL-TIME INFERENCE LOOP
# =============================================================================
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    current_time = time.time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results   = detector.detect(mp_image)

    state_text = "WAITING FOR DATA..."
    color      = (255, 255, 255)

    if results.face_landmarks:
        missing_face_frames = 0

        h, w, _ = frame.shape
        landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in results.face_landmarks[0]])

        # --- A. EXTRACT & SMOOTH ---
        raw_ear = (calculate_aspect_ratio(landmarks, LEFT_EYE_IDX) +
                   calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)) / 2.0
        ear = ear_kf.update(raw_ear)
        mar = calculate_aspect_ratio(landmarks, MOUTH_IDX)

        raw_pitch, yaw, roll = get_head_pose(landmarks, w, h)
        pitch = pitch_kf.update(raw_pitch) - baseline_pitch
        pitch = float(np.clip(pitch, -90, 90))
        yaw   = float(np.clip(yaw,   -90, 90))
        roll  = float(np.clip(roll,  -90, 90))

        # --- B. SESSION-ADAPTIVE THRESHOLDS ---
        # Sensitivity increases progressively as session gets longer.
        session_minutes  = (current_time - session_start_time) / 60.0
        fatigue_factor   = min(1.0, session_minutes / 60.0)  # ramps from 0→1 over first hour
        ear_closed_thresh = min(EAR_CLOSED_MAX,      BASE_EAR_CLOSED  + fatigue_factor * ADAPT_EAR_RATE)
        vote_threshold    = max(VOTE_THRESHOLD_MIN,  BASE_VOTE_THRESHOLD - fatigue_factor * ADAPT_VOTE_RATE)

        # --- C. NORMALIZE ---
        norm_ear = ear / baseline_ear
        ear_history.append(norm_ear)
        mar_history.append(mar)

        # --- D. EVENT LOGIC ---
        current_active_blink_dur = 0.0
        if norm_ear < ear_closed_thresh:
            if not is_blinking:
                is_blinking      = True
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

        # --- E. PREDICT ---
        if len(ear_history) == WINDOW_SIZE:

            if window_filled_time is None:
                window_filled_time = current_time

            # Core features
            perclos       = np.mean(np.array(ear_history) < ear_closed_thresh)
            avg_ear_val   = np.mean(ear_history)
            ear_std       = np.std(ear_history)
            avg_mar_val   = np.mean(mar_history)
            blink_rate    = len([e for e in ear_history if e < ear_closed_thresh]) / (WINDOW_SIZE / 20.0)
            avg_blink_dur = np.mean(blink_durations) if blink_durations else 0.0
            final_blink_dur = max(avg_blink_dur, current_active_blink_dur)
            recent_yawns  = len([t for t in yawn_timestamps if t > current_time - 3.0])

            # EAR slope: negative = eyes closing over the window
            ear_slope = np.polyfit(range(WINDOW_SIZE), list(ear_history), 1)[0]
            onset_detected = (ear_slope < SLOPE_ONSET_THRESHOLD) and (avg_ear_val > ear_closed_thresh)

            features = np.array([[perclos, avg_ear_val, ear_std, avg_mar_val,
                                   recent_yawns, blink_rate, final_blink_dur,
                                   pitch, yaw, roll]])

            try:
                prediction_probs = model(features, training=False).numpy()[0]
                raw_class_idx    = np.argmax(prediction_probs)
                confidence       = prediction_probs[raw_class_idx]
            except Exception as e:
                print(f"\n[!!!] FATAL AI CRASH: {e}\n")
                break

            # --- F. PERIODIC BASELINE RECALIBRATION ---
            # Quietly update baseline_ear using high-confidence attentive frames.
            if raw_class_idx == 0 and confidence > 0.82:
                confident_att_ears.append(ear)

            if (current_time - last_recalibrate_time > RECALIBRATE_INTERVAL and
                    len(confident_att_ears) >= RECALIBRATE_MIN_SAMPLES):
                candidate = np.mean(confident_att_ears)
                drift = abs(candidate - original_baseline_ear) / original_baseline_ear
                if drift <= RECALIBRATE_MAX_DRIFT:
                    baseline_ear = candidate
                    confident_att_ears.clear()
                    last_recalibrate_time = current_time
                    print(f"[RECAL] Baseline EAR updated: {baseline_ear:.3f} (drift {drift*100:.1f}%)")
                else:
                    print(f"[RECAL] Skipped — drift too large ({drift*100:.1f}%)")
                    last_recalibrate_time = current_time

            # --- G. WARMUP ---
            warmup_elapsed = current_time - window_filled_time
            if warmup_elapsed < WARMUP_SECONDS:
                state_text = f"WARMING UP... ({int(WARMUP_SECONDS - warmup_elapsed)}s)"
                color = (255, 255, 0)
                print(f"[WARMUP] {int(WARMUP_SECONDS - warmup_elapsed)}s | EAR: {avg_ear_val:.2f} | PERCLOS: {perclos:.2f}")
            else:
                # --- H. EFFECTIVE VOTE THRESHOLD ---
                # Lower it based on: alert history + EAR onset slope
                while alert_timestamps and alert_timestamps[0] < current_time - ALERT_HISTORY_WINDOW:
                    alert_timestamps.popleft()
                recent_alert_count = len(alert_timestamps)

                alert_boost = min(ALERT_BOOST_MAX, recent_alert_count * ALERT_BOOST_PER_ALERT)
                slope_boost = SLOPE_BOOST if onset_detected else 0.0
                effective_vote_threshold = max(VOTE_THRESHOLD_MIN, vote_threshold - alert_boost - slope_boost)

                # --- I. DEBOUNCING ---
                prediction_history.append((current_time, raw_class_idx))
                while prediction_history and prediction_history[0][0] < current_time - SMOOTHING_SECONDS:
                    prediction_history.popleft()

                tired_ratio = 0.0
                smoothed_class_idx = 0
                if len(prediction_history) > 0:
                    tired_votes  = sum(v[1] for v in prediction_history)
                    tired_ratio  = tired_votes / len(prediction_history)
                    smoothed_class_idx = 1 if tired_ratio >= effective_vote_threshold else 0

                # --- J. DISPLAY ---
                print(f"EAR: {avg_ear_val:.2f} | PERCLOS: {perclos:.2f} | Slope: {ear_slope:.4f}"
                      f" | BlinkDur: {final_blink_dur:.2f} | Pitch: {pitch:.1f}"
                      f" | Ratio: {tired_ratio*100:.0f}% | VoteT: {effective_vote_threshold:.2f}"
                      f" | Sess: {session_minutes:.1f}min")

                # --- K. HYBRID OVERRIDE + ALERT REGISTRATION ---
                if final_blink_dur > 2.5 or perclos > 0.85:
                    state_text = "!!! MICRO-SLEEP DETECTED !!!"
                    color = (0, 0, 255)
                    alert_timestamps.append(current_time)
                elif onset_detected and smoothed_class_idx == 0:
                    # Slope is falling but model not yet sure — stay attentive but flag
                    state_text = f"ATTENTIVE ({int(confidence*100)}%) [ONSET?]"
                    color = (0, 200, 100)
                elif smoothed_class_idx == 0:
                    state_text = f"ATTENTIVE ({int(confidence*100)}%)"
                    color = (0, 255, 0)
                else:
                    state_text = f"TIRED ({int(confidence*100)}%)"
                    color = (0, 0, 255)
                    if confidence > 0.8:
                        state_text = "!!! WAKE UP !!!"
                    alert_timestamps.append(current_time)

                if pitch < -15:
                    state_text += " [HEAD NOD]"

    else:
        missing_face_frames += 1
        if missing_face_frames > 15:
            state_text = "!!! FACE LOST / HEAD DOWN !!!"
            color = (0, 0, 255)

    # --- OVERLAY ---
    session_minutes_disp = (current_time - session_start_time) / 60.0 if session_start_time else 0
    cv2.putText(frame, state_text, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.putText(frame, f"Session: {session_minutes_disp:.1f}min", (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.imshow('Project Aegis V3', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
