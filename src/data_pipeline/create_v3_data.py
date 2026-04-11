import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import mediapipe as mp
import numpy as np
import csv
import concurrent.futures
import multiprocessing
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.feature_utils import (
    calculate_aspect_ratio,
    LEFT_EYE_IDX, RIGHT_EYE_IDX, MOUTH_IDX
)

# -----------------------------------------------------------
# 1. BATCH CONFIGURATION
# -----------------------------------------------------------
UTA_DIR = r'data\external\UTA-RLDD-RAW'
YAWDD_DIR = r'data\external\YawDD'
OUTPUT_DIR = r'data\processed\v3_chunks'

FPS = 30
WINDOW_30S = 30 * FPS
WINDOW_10S = 10 * FPS
SAMPLE_RATE = 15  # Extract 1 row every half-second

FIXED_BASELINE_EAR = 0.30
EAR_THRESH = 0.50
MAR_THRESH = 0.80

HEADERS = [
    'perclos_10s', 'perclos_30s', 'norm_avg_ear', 'ear_std', 'ear_min',
    'blink_rate', 'avg_blink_dur', 'max_blink_dur', 'time_since_last_blink', 'ear_derivative_mean',
    'avg_mar', 'max_mar', 'yawn_count', 'yawn_duration_avg',
    'pitch', 'yaw', 'roll', 'pitch_velocity', 'pitch_variance', 'target'
]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------
# 2. MATH FUNCTIONS (Biologically Bounded)
# -----------------------------------------------------------
def get_head_pose_stable(landmarks, img_w, img_h):
    face_3d = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ], dtype=np.float64)

    face_2d = np.array([
        landmarks[1], landmarks[152], landmarks[33], 
        landmarks[263], landmarks[61], landmarks[291]
    ], dtype=np.float64)

    focal_length = 1.0 * img_w
    cam_matrix = np.array([[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)
    if not success: return 0.0, 0.0, 0.0
    
    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    
    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]
    
    def normalize(angle):
        while angle > 180: angle -= 360
        while angle < -180: angle += 360
        return angle
        
    pitch = normalize(pitch)
    yaw = normalize(yaw)
    roll = normalize(roll)
    
    pitch = float(np.clip(pitch, -45.0, 30.0))  
    yaw = float(np.clip(yaw, -60.0, 60.0))      
    roll = float(np.clip(roll, -40.0, 40.0))    
    
    return pitch, yaw, roll

# -----------------------------------------------------------
# 3. CORE EXTRACTION ENGINE
# -----------------------------------------------------------
def process_video_to_chunk(video_path, target_label, chunk_filename):
    chunk_path = os.path.join(OUTPUT_DIR, chunk_filename)
    
    if os.path.exists(chunk_path):
        print(f"  [SKIPPING] {chunk_filename} already exists.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [ERROR] Cannot open {video_path}")
        return

    base_options = python.BaseOptions(model_asset_path='models/face_landmarker.task')
    options = vision.FaceLandmarkerOptions(
        base_options=base_options, 
        running_mode=vision.RunningMode.IMAGE, 
        num_faces=1
    )
    
    try:
        detector = vision.FaceLandmarker.create_from_options(options)
    except Exception as e:
        print(f"  [ERROR] Could not load FaceLandmarker. Ensure 'models/face_landmarker.task' exists. {e}")
        return

    ear_queue = deque(maxlen=WINDOW_30S)
    mar_queue = deque(maxlen=WINDOW_30S)
    pitch_queue = deque(maxlen=WINDOW_30S)
    blink_log = deque()
    yawn_log = deque()

    frame_count = 0
    is_blinking = False; blink_start_frame = 0; last_blink_frame = 0
    is_yawning = False; yawn_start_frame = 0
    
    with open(chunk_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(HEADERS)
        rows_written = 0

        while True:
            success, frame = cap.read()
            if not success: break
            frame_count += 1

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            results = detector.detect(mp_image)

            if results.face_landmarks:
                h, w, _ = frame.shape
                landmarks = np.array([(int(lm.x * w), int(lm.y * h)) for lm in results.face_landmarks[0]])

                raw_ear = (calculate_aspect_ratio(landmarks, LEFT_EYE_IDX) + calculate_aspect_ratio(landmarks, RIGHT_EYE_IDX)) / 2.0
                mar = calculate_aspect_ratio(landmarks, MOUTH_IDX)
                pitch, yaw, roll = get_head_pose_stable(landmarks, w, h)

                norm_ear = raw_ear / FIXED_BASELINE_EAR
                ear_queue.append(norm_ear)
                mar_queue.append(mar)
                pitch_queue.append(pitch)

                if norm_ear < EAR_THRESH:
                    if not is_blinking: is_blinking = True; blink_start_frame = frame_count
                else:
                    if is_blinking:
                        is_blinking = False
                        blink_log.append((frame_count, (frame_count - blink_start_frame) / FPS))
                        last_blink_frame = frame_count

                if mar > MAR_THRESH:
                    if not is_yawning: is_yawning = True; yawn_start_frame = frame_count
                else:
                    if is_yawning:
                        is_yawning = False
                        yawn_log.append((frame_count, (frame_count - yawn_start_frame) / FPS))

                while blink_log and blink_log[0][0] < frame_count - WINDOW_30S: blink_log.popleft()
                while yawn_log and yawn_log[0][0] < frame_count - WINDOW_30S: yawn_log.popleft()

                if len(ear_queue) == WINDOW_30S and frame_count % SAMPLE_RATE == 0:
                    ear_10s = list(ear_queue)[-WINDOW_10S:]
                    perclos_10s = np.mean(np.array(ear_10s) < EAR_THRESH)
                    perclos_30s = np.mean(np.array(ear_queue) < EAR_THRESH)
                    
                    b_dur = [b[1] for b in blink_log]; y_dur = [y[1] for y in yawn_log]
                    
                    writer.writerow([
                        perclos_10s, perclos_30s, np.mean(ear_queue), np.std(ear_queue), np.min(ear_queue),
                        len(blink_log) / 0.5, np.mean(b_dur) if b_dur else 0, np.max(b_dur) if b_dur else 0, 
                        (frame_count - last_blink_frame) / FPS, np.mean(np.diff(ear_queue)),
                        np.mean(mar_queue), np.max(mar_queue), len(yawn_log), np.mean(y_dur) if y_dur else 0,
                        pitch, yaw, roll, pitch_queue[-1] - pitch_queue[-2], np.var(pitch_queue), target_label
                    ])
                    rows_written += 1

    cap.release()
    print(f"  [SUCCESS] Created {chunk_filename} ({rows_written} rows extracted).")

# -----------------------------------------------------------
# 4. DATASET TRAVERSAL (MULTI-CORE PARALLEL PROCESSING)
# -----------------------------------------------------------
if __name__ == "__main__":
    print("Project Aegis V3 - Multi-Core Heavy Batch Processor Started\n")
    
    processing_jobs = []

    # 1. SCAN UTA-RLDD
    print("Scanning UTA-RLDD Dataset...")
    if os.path.exists(UTA_DIR):
        for subject_folder in os.listdir(UTA_DIR):
            subject_path = os.path.join(UTA_DIR, subject_folder)
            if not os.path.isdir(subject_path): continue
            
            for file in os.listdir(subject_path):
                file_lower = file.lower()
                full_vid_path = os.path.join(subject_path, file)
                
                if file_lower in ['0.mov', '0.mp4', '0.avi']:
                    processing_jobs.append((full_vid_path, 0, f"UTA_{subject_folder}_Attentive.csv"))
                elif file_lower in ['10.mov', '10.mp4', '10.avi']:
                    processing_jobs.append((full_vid_path, 1, f"UTA_{subject_folder}_Drowsy.csv"))
    else:
        print(f"  [ERROR] Could not find {UTA_DIR}")

    # 2. SCAN YAWDD (MIRROR ONLY - DEEP SCAN)
    print("Scanning YawDD Dataset (Mirror Only)...")
    if os.path.exists(YAWDD_DIR):
        for root, dirs, files in os.walk(YAWDD_DIR):
            # Only process if 'mirror' is somewhere in the folder path (avoids Dash)
            if 'mirror' not in root.lower():
                continue
                
            for file in files:
                file_lower = file.lower()
                if not file_lower.endswith(('.avi', '.mp4')): continue
                
                full_vid_path = os.path.join(root, file)
                
                # Match the exact text in the filenames
                if 'yawn' in file_lower:
                    processing_jobs.append((full_vid_path, 1, f"YAWDD_{file}.csv"))
                elif 'normal' in file_lower or 'talking' in file_lower:
                    processing_jobs.append((full_vid_path, 0, f"YAWDD_{file}.csv"))
    else:
        print(f"  [ERROR] YawDD folder not found at {YAWDD_DIR}")

    print(f"\nTotal Videos Queued for Processing: {len(processing_jobs)}")
    
    # Core assignment (Change to `// 2` if you need to use your laptop while it runs)
    available_cores = max(1, multiprocessing.cpu_count() - 1) 
    print(f"Firing up {available_cores} parallel CPU cores... Brace for speed.\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=available_cores) as executor:
        futures = [
            executor.submit(process_video_to_chunk, job[0], job[1], job[2]) 
            for job in processing_jobs
        ]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"  [THREAD ERROR] A parallel job failed: {e}")
                
    print("\nALL CORES FINISHED! Batch Processing Complete.")