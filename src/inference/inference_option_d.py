"""
Option D — Dual-Track State Machine
ENTER: short 2.0s voting window with 55% tired threshold (fast alert).
EXIT: count consecutive ATTENTIVE model predictions (no ratio window).
      Any tired prediction or yawn resets the streak.
      Exits TIRED only after ATTENTIVE_STREAK_NEEDED clean frames in a row.
Yawn with eyes open → immediately enters TIRED and resets the exit streak.
"""
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

class KalmanFilter1D:
    def __init__(self, process_variance=1e-4, measurement_variance=1e-2):
        self.Q = process_variance
        self.R = measurement_variance
        self.P = 1.0
        self.x = None

    def update(self, z):
        if self.x is None:
            self.x = z
            return self.x
        self.P  = self.P + self.Q
        K       = self.P / (self.P + self.R)
        self.x  = self.x + K * (z - self.x)
        self.P  = (1 - K) * self.P
        return self.x


# =============================================================================
# PARAMETERS
# =============================================================================
WINDOW_SIZE          = 30
CALIBRATION_FRAMES   = 50
WARMUP_SECONDS       = 5.0
EAR_CLOSED_THRESHOLD = 0.75

ENTER_WINDOW    = 2.0   # seconds — short window for fast alert
ENTER_THRESHOLD = 0.55  # tired_ratio in ENTER_WINDOW to enter TIRED

# To EXIT tired, need this many consecutive attentive predictions
# At ~15fps, 20 frames ≈ 1.3s of clean alertness
ATTENTIVE_STREAK_NEEDED = 20

YAWN_OPEN_EAR_MIN = 0.80

# =============================================================================
# SETUP
# =============================================================================
model = tf.keras.models.load_model('models/vision_model.h5')
print("Model loaded.")

base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
)

ear_kf   = KalmanFilter1D(process_variance=1e-4, measurement_variance=1e-2)
pitch_kf = KalmanFilter1D(process_variance=0.5,  measurement_variance=2.0)
yaw_kf   = KalmanFilter1D(process_variance=0.5,  measurement_variance=2.0)
roll_kf  = KalmanFilter1D(process_variance=0.5,  measurement_variance=2.0)

ear_history        = deque(maxlen=WINDOW_SIZE)
ear_timestamps     = deque(maxlen=WINDOW_SIZE)
mar_history        = deque(maxlen=WINDOW_SIZE)
blink_durations    = deque(maxlen=10)
yawn_timestamps    = deque(maxlen=10)
prediction_history = deque()   # enter-track only (ENTER_WINDOW seconds)

blink_start_time    = None
is_blinking         = False
prev_is_yawning     = False
is_yawning          = False
missing_face_frames = 0
window_filled_time  = None
alert_state         = 0
attentive_streak    = 0   # consecutive attentive frames (exit track)

cap = cv2.VideoCapture(0)

# =============================================================================
# CALIBRATION
# =============================================================================
print("CALIBRATING — look straight at the camera, relax face.")
calibration_ears, calibration_pitches = [], []

while len(calibration_ears) < CALIBRATION_FRAMES and cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
    if results.face_landmarks:
        h, w, _ = frame.shape
        lm = np.array([(int(p.x * w), int(p.y * h)) for p in results.face_landmarks[0]])
        ear = (calculate_aspect_ratio(lm, LEFT_EYE_IDX) +
               calculate_aspect_ratio(lm, RIGHT_EYE_IDX)) / 2.0
        calibration_ears.append(ear)
        pitch, _, _ = get_head_pose(lm, w, h)
        calibration_pitches.append(pitch)
    cv2.putText(frame, f"CALIBRATING ({len(calibration_ears)}/{CALIBRATION_FRAMES})",
                (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.imshow("Project Aegis — Option D (Dual Track)", frame)
    cv2.waitKey(1)

baseline_ear   = np.mean(calibration_ears)   if calibration_ears   else 0.30
baseline_pitch = np.mean(calibration_pitches) if calibration_pitches else 0.0
print(f"Baseline EAR: {baseline_ear:.3f} | Baseline Pitch: {baseline_pitch:.1f}")

# =============================================================================
# INFERENCE LOOP
# =============================================================================
print("Running. Press Q to quit.")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    current_time = time.time()
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    state_text = "WAITING FOR DATA..."
    diag_text  = ""
    color      = (255, 255, 255)

    if results.face_landmarks:
        missing_face_frames = 0
        h, w, _ = frame.shape
        lm = np.array([(int(p.x * w), int(p.y * h)) for p in results.face_landmarks[0]])

        raw_ear = (calculate_aspect_ratio(lm, LEFT_EYE_IDX) +
                   calculate_aspect_ratio(lm, RIGHT_EYE_IDX)) / 2.0
        ear = ear_kf.update(raw_ear)
        mar = calculate_aspect_ratio(lm, MOUTH_IDX)

        raw_pitch, raw_yaw, raw_roll = get_head_pose(lm, w, h)
        pitch = float(np.clip(pitch_kf.update(raw_pitch) - baseline_pitch, -90, 90))
        yaw   = float(np.clip(yaw_kf.update(raw_yaw),  -90, 90))
        roll  = float(np.clip(roll_kf.update(raw_roll), -90, 90))

        norm_ear = ear / baseline_ear
        ear_history.append(norm_ear)
        ear_timestamps.append(current_time)
        mar_history.append(mar)

        # --- blink tracking ---
        current_active_blink_dur = 0.0
        if norm_ear < EAR_CLOSED_THRESHOLD:
            if not is_blinking:
                is_blinking      = True
                blink_start_time = current_time
            current_active_blink_dur = current_time - blink_start_time
        else:
            if is_blinking:
                is_blinking = False
                blink_durations.append(current_time - blink_start_time)

        # --- yawn tracking (rising edge) ---
        prev_is_yawning = is_yawning
        if mar > 0.55:
            if not is_yawning:
                is_yawning = True
                yawn_timestamps.append(current_time)
        else:
            is_yawning = False

        yawn_open_just_detected = (is_yawning and not prev_is_yawning
                                   and norm_ear > YAWN_OPEN_EAR_MIN)

        if len(ear_history) == WINDOW_SIZE:
            if window_filled_time is None:
                window_filled_time = current_time

            window_duration = ear_timestamps[-1] - ear_timestamps[0]
            if window_duration <= 0:
                window_duration = 1.0

            ear_arr     = np.array(ear_history)
            closed_mask = ear_arr < EAR_CLOSED_THRESHOLD
            perclos       = float(np.mean(closed_mask))
            avg_ear_val   = float(np.mean(ear_arr))
            ear_std       = float(np.std(ear_arr))
            avg_mar_val   = float(np.mean(mar_history))
            avg_blink_dur = float(np.mean(blink_durations)) if blink_durations else 0.0
            final_blink_dur = max(avg_blink_dur, current_active_blink_dur)
            recent_yawns  = len([t for t in yawn_timestamps if t > current_time - 3.0])
            blink_rate    = float(np.sum(closed_mask)) / window_duration

            features = np.array([[perclos, avg_ear_val, ear_std, avg_mar_val,
                                   recent_yawns, blink_rate, final_blink_dur,
                                   pitch, yaw, roll]])

            try:
                probs = model(features, training=False).numpy()[0]
                raw_class_idx = int(np.argmax(probs))
                confidence    = probs[raw_class_idx]
            except Exception as e:
                print(f"[CRASH] {e}")
                break

            if perclos < 0.05 and avg_ear_val > 0.97:
                raw_class_idx = 0
            if mar > 0.75:
                raw_class_idx = 1

            warmup_elapsed = current_time - window_filled_time
            if warmup_elapsed < WARMUP_SECONDS:
                state_text = f"WARMING UP... ({int(WARMUP_SECONDS - warmup_elapsed)}s)"
                color = (255, 255, 0)
            else:
                # --- DUAL TRACK STATE MACHINE ---

                # ENTER TRACK: short rolling window
                prediction_history.append((current_time, raw_class_idx))
                while prediction_history and prediction_history[0][0] < current_time - ENTER_WINDOW:
                    prediction_history.popleft()
                enter_ratio = (sum(v for _, v in prediction_history) /
                               max(len(prediction_history), 1))

                # EXIT TRACK: consecutive attentive streak
                if raw_class_idx == 0:
                    attentive_streak += 1
                else:
                    attentive_streak = 0  # any tired vote breaks the streak

                # Yawn with eyes open → force TIRED immediately, wipe streak
                if yawn_open_just_detected:
                    alert_state = 1
                    attentive_streak = 0
                    print("[YAWN+EYES OPEN] Forced TIRED, streak reset")

                # State transitions
                if alert_state == 0 and enter_ratio >= ENTER_THRESHOLD:
                    alert_state = 1
                    attentive_streak = 0
                elif alert_state == 1 and attentive_streak >= ATTENTIVE_STREAK_NEEDED:
                    alert_state = 0
                    prediction_history.clear()
                    attentive_streak = 0

                diag_text = (f"Enter: {enter_ratio*100:.0f}%  "
                             f"Streak: {attentive_streak}/{ATTENTIVE_STREAK_NEEDED}  "
                             f"EAR: {avg_ear_val:.2f}")
                print(f"EAR:{avg_ear_val:.2f} PERCLOS:{perclos:.2f} "
                      f"Enter:{enter_ratio*100:.0f}% Streak:{attentive_streak}")

                if final_blink_dur > 2.5 or perclos > 0.85:
                    state_text = "!!! MICRO-SLEEP !!!"
                    color = (0, 0, 255)
                elif alert_state == 0:
                    state_text = f"ATTENTIVE ({int(confidence*100)}%)"
                    color = (0, 255, 0)
                else:
                    state_text = f"TIRED ({int(confidence*100)}%)"
                    color = (0, 0, 255)
                    if confidence > 0.8:
                        state_text = "!!! WAKE UP !!!"

                if pitch < -15:
                    state_text += " [HEAD NOD]"
    else:
        missing_face_frames += 1
        if missing_face_frames > 15:
            state_text = "!!! FACE LOST !!!"
            color = (0, 0, 255)

    cv2.putText(frame, "[OPT-D] Dual-Track State Machine",
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(frame, state_text,
                (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    if diag_text:
        cv2.putText(frame, diag_text,
                    (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 0), 1)

    # Streak bar (visual) — fills toward ATTENTIVE_STREAK_NEEDED
    if alert_state == 1:
        bar_w = int((attentive_streak / ATTENTIVE_STREAK_NEEDED) * 300)
        cv2.rectangle(frame, (10, 125), (310, 145), (50, 50, 50), -1)
        if bar_w > 0:
            cv2.rectangle(frame, (10, 125), (10 + bar_w, 145), (0, 200, 100), -1)
        cv2.putText(frame, f"Recovery streak: {attentive_streak}/{ATTENTIVE_STREAK_NEEDED}",
                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 100), 1)

    cv2.imshow("Project Aegis — Option D (Dual Track)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
