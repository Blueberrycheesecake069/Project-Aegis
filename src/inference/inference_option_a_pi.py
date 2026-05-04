"""
Option A (Pi) — Asymmetric Windows
ENTER uses a short 2.5s window (fast alert).
EXIT uses a longer 4.0s window with a higher tired threshold (easier to leave TIRED).
Yawn with eyes open immediately forces TIRED state.
"""
import sys
import os

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR      = os.path.dirname(_SCRIPT_DIR)
_PROJECT_ROOT = os.path.dirname(_SRC_DIR)
sys.path.insert(0, _SRC_DIR)

import cv2
import mediapipe as mp
import numpy as np
import time
import queue
import subprocess
import threading
from collections import deque
from flask import Flask, Response
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from utils.feature_utils_pi import (
    calculate_aspect_ratio, get_head_pose,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX
)

# =============================================================================
# FLASK MJPEG STREAM
# =============================================================================
_stream_queue = queue.Queue(maxsize=1)
app = Flask(__name__)

@app.route('/')
def index():
    return '<html><body><img src="/video_feed" width="640"></body></html>'

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            frame = _stream_queue.get()
            _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

threading.Thread(
    target=lambda: app.run(host='0.0.0.0', port=8080, threaded=True, use_reloader=False),
    daemon=True
).start()
print("Stream live at http://<pi-ip>:8080")

# =============================================================================
# AUDIO ALERT
# =============================================================================
ALERT_AUDIO_PATH = os.path.join(_PROJECT_ROOT, 'assets', 'alert.wav')
ALERT_COOLDOWN   = 8.0
_last_alert_time = 0.0

def _play_alert_worker():
    ext = os.path.splitext(ALERT_AUDIO_PATH)[1].lower()
    try:
        if ext == '.mp3':
            subprocess.run(['mpg123', '-q', ALERT_AUDIO_PATH], check=False)
        else:
            subprocess.run(['aplay', '-q', ALERT_AUDIO_PATH], check=False)
    except FileNotFoundError:
        pass

def trigger_alert():
    global _last_alert_time
    now = time.time()
    if now - _last_alert_time >= ALERT_COOLDOWN:
        _last_alert_time = now
        threading.Thread(target=_play_alert_worker, daemon=True).start()

# =============================================================================
# TFLITE
# =============================================================================
try:
    from tflite_runtime.interpreter import Interpreter as TFLiteInterpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter as TFLiteInterpreter

# =============================================================================
# KALMAN FILTER
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
# LOAD MODEL & LANDMARKER
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
    print(f"Error loading TFLite model: {e}")
    sys.exit(1)

base_options = python.BaseOptions(model_asset_path=_LANDMARKER_PATH)
detector = vision.FaceLandmarker.create_from_options(
    vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_faces=1
    )
)

# =============================================================================
# PARAMETERS
# =============================================================================
WINDOW_SIZE          = 30
CALIBRATION_FRAMES   = 50
MAX_CALIB_ATTEMPTS   = 3
CALIB_EAR_MIN        = 0.20
CALIB_EAR_MAX        = 0.42
WARMUP_SECONDS       = 5.0
EAR_CLOSED_THRESHOLD = 0.75

ENTER_WINDOW    = 2.5   # seconds — short window for fast alert
EXIT_WINDOW     = 4.0   # seconds — longer window, easier to exit
ENTER_THRESHOLD = 0.55  # tired_ratio in ENTER_WINDOW to enter TIRED
EXIT_THRESHOLD  = 0.35  # tired_ratio in EXIT_WINDOW must drop below this to exit TIRED

YAWN_OPEN_EAR_MIN = 0.80

# =============================================================================
# STATE
# =============================================================================
ear_history        = deque(maxlen=WINDOW_SIZE)
ear_timestamps     = deque(maxlen=WINDOW_SIZE)
mar_history        = deque(maxlen=WINDOW_SIZE)
blink_durations    = deque(maxlen=10)
yawn_timestamps    = deque(maxlen=10)
prediction_history = deque()

blink_start_time    = None
is_blinking         = False
prev_is_yawning     = False
is_yawning          = False
missing_face_frames = 0
window_filled_time  = None
alert_state         = 0

ear_kf   = KalmanFilter1D(process_variance=1e-4, measurement_variance=1e-2)
pitch_kf = KalmanFilter1D(process_variance=0.5,  measurement_variance=2.0)
yaw_kf   = KalmanFilter1D(process_variance=0.5,  measurement_variance=2.0)
roll_kf  = KalmanFilter1D(process_variance=0.5,  measurement_variance=2.0)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =============================================================================
# CALIBRATION
# =============================================================================
def _push_frame(frame):
    if _stream_queue.full():
        try:
            _stream_queue.get_nowait()
        except queue.Empty:
            pass
    _stream_queue.put(frame)

def run_calibration(cap, detector, retry_msg=None):
    calibration_ears, calibration_pitches = [], []
    while len(calibration_ears) < CALIBRATION_FRAMES and cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if results.face_landmarks:
            h, w, _ = frame.shape
            lm = np.array([(int(p.x * w), int(p.y * h)) for p in results.face_landmarks[0]])
            ear = (calculate_aspect_ratio(lm, LEFT_EYE_IDX) +
                   calculate_aspect_ratio(lm, RIGHT_EYE_IDX)) / 2.0
            calibration_ears.append(ear)
            pitch, _, _ = get_head_pose(lm, w, h)
            calibration_pitches.append(pitch)
        cv2.putText(frame, f"CALIBRATING... ({len(calibration_ears)}/{CALIBRATION_FRAMES})",
                    (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, "RELAX FACE — LOOK STRAIGHT AT CAMERA",
                    (50, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        if retry_msg:
            cv2.putText(frame, retry_msg, (50, 320),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
        _push_frame(frame)
    baseline_ear   = np.mean(calibration_ears)   if calibration_ears   else 0.0
    baseline_pitch = np.mean(calibration_pitches) if calibration_pitches else 0.0
    return baseline_ear, baseline_pitch

print("Starting calibration...")
baseline_ear, baseline_pitch = 0.0, 0.0

for attempt in range(MAX_CALIB_ATTEMPTS):
    retry_msg = (f"BAD CALIBRATION — RETRY {attempt+1}/{MAX_CALIB_ATTEMPTS}"
                 if attempt > 0 else None)
    baseline_ear, baseline_pitch = run_calibration(cap, detector, retry_msg)

    if CALIB_EAR_MIN <= baseline_ear <= CALIB_EAR_MAX:
        print(f"Calibration OK. EAR: {baseline_ear:.3f} | Pitch: {baseline_pitch:.1f}")
        break
    else:
        print(f"[CALIB FAIL] EAR={baseline_ear:.3f} outside [{CALIB_EAR_MIN}, {CALIB_EAR_MAX}].")
        if attempt == MAX_CALIB_ATTEMPTS - 1:
            baseline_ear = 0.30
            print("[CALIB] Fallback EAR=0.30 applied.")
        else:
            deadline = time.time() + 2.5
            while time.time() < deadline and cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                cv2.putText(frame, f"EAR={baseline_ear:.3f} out of range.",
                            (50, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
                cv2.putText(frame, "Sit straight, look at camera, don't blink.",
                            (50, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                _push_frame(frame)

# =============================================================================
# INFERENCE LOOP
# =============================================================================
print("Starting inference loop.")

while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    current_time = time.time()
    rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

    state_text = "WAITING FOR DATA..."
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
                interpreter.set_tensor(_input_details[0]['index'],
                                       features.astype(np.float32))
                interpreter.invoke()
                probs = interpreter.get_tensor(_output_details[0]['index'])[0]
                raw_class_idx = int(np.argmax(probs))
                confidence    = probs[raw_class_idx]
            except Exception as e:
                print(f"[FATAL] {e}")
                break

            if perclos < 0.05 and avg_ear_val > 0.97:
                raw_class_idx = 0
            if mar > 0.75:
                raw_class_idx = 1

            warmup_elapsed = current_time - window_filled_time
            if warmup_elapsed < WARMUP_SECONDS:
                state_text = f"WARMING UP... ({int(WARMUP_SECONDS - warmup_elapsed)}s)"
                color = (255, 255, 0)
                print(f"[WARMUP] {int(WARMUP_SECONDS - warmup_elapsed)}s | "
                      f"EAR:{avg_ear_val:.2f} PERCLOS:{perclos:.2f}")
            else:
                # --- ASYMMETRIC DEBOUNCING ---
                prediction_history.append((current_time, raw_class_idx))
                while prediction_history and prediction_history[0][0] < current_time - EXIT_WINDOW:
                    prediction_history.popleft()

                enter_votes = [(t, v) for t, v in prediction_history
                               if t > current_time - ENTER_WINDOW]
                exit_votes  = list(prediction_history)

                enter_ratio = sum(v for _, v in enter_votes) / max(len(enter_votes), 1)
                exit_ratio  = sum(v for _, v in exit_votes)  / max(len(exit_votes),  1)

                if yawn_open_just_detected:
                    alert_state = 1
                    print("[YAWN+EYES OPEN] Forced TIRED")

                if alert_state == 0 and enter_ratio >= ENTER_THRESHOLD:
                    alert_state = 1
                elif alert_state == 1 and exit_ratio < EXIT_THRESHOLD:
                    alert_state = 0
                    prediction_history.clear()

                print(f"EAR:{avg_ear_val:.2f} PERCLOS:{perclos:.2f} "
                      f"Enter:{enter_ratio*100:.0f}% Exit:{exit_ratio*100:.0f}% "
                      f"WinDur:{window_duration:.2f}s")

                if final_blink_dur > 2.5 or perclos > 0.85:
                    state_text = "!!! MICRO-SLEEP DETECTED !!!"
                    color = (0, 0, 255)
                    trigger_alert()
                elif alert_state == 0:
                    state_text = f"ATTENTIVE ({int(confidence*100)}%)"
                    color = (0, 255, 0)
                else:
                    state_text = f"TIRED ({int(confidence*100)}%)"
                    color = (0, 0, 255)
                    trigger_alert()
                    if confidence > 0.8:
                        state_text = "!!! WAKE UP !!!"

                if pitch < -15:
                    state_text += " [HEAD NOD]"
    else:
        missing_face_frames += 1
        if missing_face_frames > 15:
            state_text = "!!! FACE LOST / HEAD DOWN !!!"
            color = (0, 0, 255)

    cv2.putText(frame, state_text, (30, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    _push_frame(frame)

cap.release()
