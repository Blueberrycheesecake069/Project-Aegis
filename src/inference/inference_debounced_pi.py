import sys
import os

# Make src/ importable regardless of where the script is launched from
_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))   # src/inference/
_SRC_DIR      = os.path.dirname(_SCRIPT_DIR)                  # src/
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)                     # project root
sys.path.insert(0, _SRC_DIR)

import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Pi-specific utils: corrected camera intrinsics (Pi Camera v2 focal length)
from utils.feature_utils_pi import (
    calculate_aspect_ratio, get_head_pose,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX
)

# TFLite runtime — lightweight package on Pi, falls back to full TF on PC
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter as TFLiteInterpreter


# =============================================================================
# KALMAN FILTER (1D)
# Smooths noisy landmark-derived signals by separating real physiological
# change from MediaPipe jitter.
# =============================================================================
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
# 1. LOAD TFLITE MODEL
# =============================================================================
_MODEL_PATH      = os.path.join(_PROJECT_ROOT, 'models', 'vision_model.tflite')
_LANDMARKER_PATH = os.path.join(_PROJECT_ROOT, 'models', 'face_landmarker.task')

try:
    interpreter = TFLiteInterpreter(model_path=_MODEL_PATH, num_threads=2)
    interpreter.allocate_tensors()
    _input_details  = interpreter.get_input_details()
    _output_details = interpreter.get_output_details()
    print(f"TFLite model loaded: {_MODEL_PATH}")
except Exception as e:
    print(f"Error: Could not load TFLite model — {e}")
    print("Run convert_to_tflite.py on your PC first.")
    sys.exit(1)

base_options = python.BaseOptions(model_asset_path=_LANDMARKER_PATH)
options      = vision.FaceLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)


# =============================================================================
# 2. PARAMETERS
# =============================================================================
WINDOW_SIZE          = 60
CALIBRATION_FRAMES   = 50
MAX_CALIB_ATTEMPTS   = 3
CALIB_EAR_MIN        = 0.20   # physiological lower bound for resting EAR
CALIB_EAR_MAX        = 0.42   # physiological upper bound for resting EAR
SMOOTHING_SECONDS    = 5.0
WARMUP_SECONDS       = 5.0
EAR_CLOSED_THRESHOLD = 0.75
TIRED_ENTER_THRESHOLD = 0.60  # tired_ratio must exceed this to trigger TIRED
TIRED_EXIT_THRESHOLD  = 0.25  # tired_ratio must drop below this to revert to ATTENTIVE


# =============================================================================
# 3. HISTORIES & STATE
# =============================================================================
ear_history    = deque(maxlen=WINDOW_SIZE)
ear_timestamps = deque(maxlen=WINDOW_SIZE)
mar_history    = deque(maxlen=WINDOW_SIZE)
blink_durations    = deque(maxlen=10)
yawn_timestamps    = deque(maxlen=10)
prediction_history = deque()

blink_start_time    = None
is_blinking         = False
is_yawning          = False
missing_face_frames = 0
window_filled_time  = None
alert_state         = 0     # 0 = attentive, 1 = tired (hysteresis state machine)
last_valid_pitch    = None  # for pitch spike rejection

# Kalman filters — pitch/yaw/roll all get smoothed now
ear_kf   = KalmanFilter1D(process_variance=1e-4, measurement_variance=1e-2)
pitch_kf = KalmanFilter1D(process_variance=0.5,  measurement_variance=2.0)
yaw_kf   = KalmanFilter1D(process_variance=0.5,  measurement_variance=2.0)
roll_kf  = KalmanFilter1D(process_variance=0.5,  measurement_variance=2.0)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


# =============================================================================
# 4. CALIBRATION PHASE (with validation + retry)
# =============================================================================
def run_calibration(cap, detector, retry_msg=None):
    """
    Collects CALIBRATION_FRAMES of EAR/pitch data from an alert driver.
    Validates that baseline_ear falls in the physiological range [0.20, 0.42].
    Returns (baseline_ear, baseline_pitch).
    """
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
            landmarks = np.array(
                [(int(lm.x * w), int(lm.y * h)) for lm in results.face_landmarks[0]]
            )
            ear = (calculate_aspect_ratio(landmarks, LEFT_EYE_IDX) +
                   calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)) / 2.0
            calibration_ears.append(ear)

            pitch, _, _ = get_head_pose(landmarks, w, h)
            calibration_pitches.append(pitch)

        progress = f"CALIBRATING... ({len(calibration_ears)}/{CALIBRATION_FRAMES})"
        cv2.putText(frame, progress, (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, "RELAX FACE — LOOK STRAIGHT AT CAMERA", (50, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        if retry_msg:
            cv2.putText(frame, retry_msg, (50, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
        cv2.imshow('Project Aegis Pi', frame)
        cv2.waitKey(1)

    baseline_ear   = np.mean(calibration_ears)   if calibration_ears    else 0.0
    baseline_pitch = np.mean(calibration_pitches) if calibration_pitches else 0.0
    return baseline_ear, baseline_pitch


print("Starting calibration...")
baseline_ear   = 0.0
baseline_pitch = 0.0

for attempt in range(MAX_CALIB_ATTEMPTS):
    retry_msg = (f"BAD CALIBRATION — RETRY {attempt + 1}/{MAX_CALIB_ATTEMPTS}"
                 if attempt > 0 else None)
    baseline_ear, baseline_pitch = run_calibration(cap, detector, retry_msg)

    if CALIB_EAR_MIN <= baseline_ear <= CALIB_EAR_MAX:
        print(f"Calibration OK. Baseline EAR: {baseline_ear:.3f} | Pitch: {baseline_pitch:.1f}")
        break
    else:
        print(f"[CALIB FAIL] Baseline EAR={baseline_ear:.3f} outside "
              f"[{CALIB_EAR_MIN}, {CALIB_EAR_MAX}]. "
              f"{'Retrying...' if attempt < MAX_CALIB_ATTEMPTS - 1 else 'Using fallback 0.30.'}")

        if attempt == MAX_CALIB_ATTEMPTS - 1:
            # Exhausted retries — use safe fallback so the system still runs
            baseline_ear = 0.30
            print("[CALIB] Fallback baseline EAR=0.30 applied.")
        else:
            # Brief pause with on-screen message before retrying
            deadline = time.time() + 2.5
            while time.time() < deadline and cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                cv2.putText(frame,
                            f"EAR={baseline_ear:.3f} out of normal range.",
                            (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
                cv2.putText(frame,
                            "Sit straight, look at camera, don't blink.",
                            (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                cv2.imshow('Project Aegis Pi', frame)
                cv2.waitKey(1)


# =============================================================================
# 5. REAL-TIME INFERENCE LOOP
# =============================================================================
print("Starting inference loop. Press Q to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    current_time = time.time()
    rgb_frame    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    results      = detector.detect(mp_image)

    state_text = "WAITING FOR DATA..."
    color      = (255, 255, 255)

    if results.face_landmarks:
        missing_face_frames = 0

        h, w, _ = frame.shape
        landmarks = np.array(
            [(int(lm.x * w), int(lm.y * h)) for lm in results.face_landmarks[0]]
        )

        # --- A. EXTRACT & SMOOTH ---
        raw_ear = (calculate_aspect_ratio(landmarks, LEFT_EYE_IDX) +
                   calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)) / 2.0
        ear = ear_kf.update(raw_ear)
        mar = calculate_aspect_ratio(landmarks, MOUTH_IDX)

        raw_pitch, raw_yaw, raw_roll = get_head_pose(landmarks, w, h)

        # Spike rejection: discard readings that jump >20° in one frame
        if last_valid_pitch is not None and abs(raw_pitch - last_valid_pitch) > 20:
            raw_pitch = last_valid_pitch
        else:
            last_valid_pitch = raw_pitch

        # All three angles are now Kalman-filtered to suppress landmark jitter
        pitch = pitch_kf.update(raw_pitch) - baseline_pitch
        yaw   = yaw_kf.update(raw_yaw)
        roll  = roll_kf.update(raw_roll)

        pitch = float(np.clip(pitch, -90, 90))
        yaw   = float(np.clip(yaw,   -90, 90))
        roll  = float(np.clip(roll,  -90, 90))

        # --- B. NORMALIZE & STORE WITH TIMESTAMP ---
        norm_ear = ear / baseline_ear
        ear_history.append(norm_ear)
        ear_timestamps.append(current_time)   # paired timestamp for FPS-independent features
        mar_history.append(mar)

        # --- C. EVENT LOGIC ---
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

        if mar > 0.55:
            if not is_yawning:
                is_yawning = True
                yawn_timestamps.append(current_time)
        else:
            is_yawning = False

        # --- D. PREDICT ---
        if len(ear_history) == WINDOW_SIZE:

            if window_filled_time is None:
                window_filled_time = current_time

            # Actual time span of the current window in seconds.
            # This replaces the hardcoded (WINDOW_SIZE / 20.0) denominator
            # and makes blink_rate correct at any FPS.
            window_duration = ear_timestamps[-1] - ear_timestamps[0]
            if window_duration <= 0:
                window_duration = 1.0   # safety fallback, should never happen

            ear_arr     = np.array(ear_history)
            closed_mask = ear_arr < EAR_CLOSED_THRESHOLD

            perclos         = float(np.mean(closed_mask))
            avg_ear_val     = float(np.mean(ear_arr))
            ear_std         = float(np.std(ear_arr))
            avg_mar_val     = float(np.mean(mar_history))
            avg_blink_dur   = float(np.mean(blink_durations)) if blink_durations else 0.0
            final_blink_dur = max(avg_blink_dur, current_active_blink_dur)
            recent_yawns    = len([t for t in yawn_timestamps if t > current_time - 3.0])

            # FPS-independent blink_rate: closed frames / actual seconds in window
            blink_rate = float(np.sum(closed_mask)) / window_duration

            features = np.array([[perclos, avg_ear_val, ear_std, avg_mar_val,
                                   recent_yawns, blink_rate, final_blink_dur,
                                   pitch, yaw, roll]])

            # --- AI INFERENCE (TFLite) ---
            try:
                interpreter.set_tensor(
                    _input_details[0]['index'],
                    features.astype(np.float32)
                )
                interpreter.invoke()
                prediction_probs = interpreter.get_tensor(_output_details[0]['index'])[0]
                raw_class_idx    = np.argmax(prediction_probs)
                confidence       = prediction_probs[raw_class_idx]
            except Exception as e:
                print(f"\n[!!!] FATAL AI CRASH: {e}\n")
                break

            # Feature override: only force attentive when eyes are very close
            # to the calibrated fully-open baseline (within 3%)
            if perclos < 0.05 and avg_ear_val > 0.97:
                raw_class_idx = 0

            # Manual yawn override: resting MAR is always ~0.66 due to broken
            # MOUTH_IDX, but a real yawn pushes it above 0.75 as the lower lip
            # drops. Force a tired vote regardless of model output.
            if mar > 0.75:
                raw_class_idx = 1

            # --- WARMUP ---
            warmup_elapsed = current_time - window_filled_time
            if warmup_elapsed < WARMUP_SECONDS:
                state_text = f"WARMING UP... ({int(WARMUP_SECONDS - warmup_elapsed)}s)"
                color = (255, 255, 0)
                print(f"[WARMUP] {int(WARMUP_SECONDS - warmup_elapsed)}s | "
                      f"EAR: {avg_ear_val:.2f} | PERCLOS: {perclos:.2f} | "
                      f"WinDur: {window_duration:.2f}s")
            else:
                # --- TIME-BASED DEBOUNCING ---
                prediction_history.append((current_time, raw_class_idx))
                while prediction_history and prediction_history[0][0] < current_time - SMOOTHING_SECONDS:
                    prediction_history.popleft()

                tired_ratio = 0.0
                if len(prediction_history) > 0:
                    tired_votes = sum(v[1] for v in prediction_history)
                    tired_ratio = tired_votes / len(prediction_history)

                # Hysteresis: high threshold to enter TIRED, low threshold to exit
                if alert_state == 0 and tired_ratio >= TIRED_ENTER_THRESHOLD:
                    alert_state = 1
                elif alert_state == 1 and tired_ratio < TIRED_EXIT_THRESHOLD:
                    alert_state = 0
                    prediction_history.clear()  # wipe old tired votes so recovery sticks
                smoothed_idx = alert_state

                print(f"EAR:{avg_ear_val:.2f} PERCLOS:{perclos:.2f} Pitch:{pitch:.1f} "
                      f"vote:{tired_ratio*100:.0f}% state:{'TIRED' if alert_state else 'ATTENTIVE'}")

                # --- HYBRID OVERRIDE ---
                if final_blink_dur > 2.5 or perclos > 0.85:
                    state_text = "!!! MICRO-SLEEP DETECTED !!!"
                    color = (0, 0, 255)
                elif smoothed_idx == 0:
                    state_text = f"ATTENTIVE ({int(confidence * 100)}%)"
                    color = (0, 255, 0)
                else:
                    state_text = f"TIRED ({int(confidence * 100)}%)"
                    color = (0, 0, 255)
                    if confidence > 0.8:
                        state_text = "!!! WAKE UP !!!"

                if pitch < -15:
                    state_text += " [HEAD NOD]"

    else:
        missing_face_frames += 1
        if missing_face_frames > 15:
            state_text = "!!! FACE LOST / HEAD DOWN !!!"
            color = (0, 0, 255)

    # --- OVERLAY ---
    cv2.putText(frame, state_text, (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    cv2.imshow('Project Aegis Pi', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
