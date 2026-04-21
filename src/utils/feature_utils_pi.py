import cv2
import numpy as np

# ---------------------------------------------------------------------
# CAMERA INTRINSICS
# Set CAMERA to match your hardware. This affects pitch/yaw/roll accuracy.
#
#   "picamera2"  — Raspberry Pi Camera Module v2 (Sony IMX219, 3.04mm lens)
#   "generic"    — fallback approximation, same as original code (focal = img_w)
# ---------------------------------------------------------------------
CAMERA = "picamera2"

# Pi Camera Module v2 physical specs
_FOCAL_MM  = 3.04   # lens focal length in mm
_SENSOR_W  = 3.68   # sensor width in mm
_SENSOR_H  = 2.76   # sensor height in mm

# ---------------------------------------------------------------------
# 1. 3D FACE MODEL (Generic Human Face)
# ---------------------------------------------------------------------
FACE_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float64)


def _build_camera_matrix(img_w, img_h):
    """
    Returns the camera intrinsics matrix based on CAMERA setting.

    For Pi Camera v2 the focal length in pixels is derived from the
    physical lens focal length and sensor dimensions:
        fx = focal_mm * pixel_cols / sensor_width_mm
        fy = focal_mm * pixel_rows / sensor_height_mm

    At 640x480 this gives fx≈529, fy≈529 — compared to the original
    code's approximation of 640, which was ~20% too high and caused
    systematic pitch/yaw/roll errors.
    """
    if CAMERA == "picamera2":
        fx = _FOCAL_MM * img_w / _SENSOR_W
        fy = _FOCAL_MM * img_h / _SENSOR_H
    else:
        # generic fallback — matches original feature_utils.py behaviour
        fx = float(img_w)
        fy = float(img_w)

    cx = img_w / 2.0
    cy = img_h / 2.0

    return np.array([[fx, 0,  cx],
                     [0,  fy, cy],
                     [0,  0,  1.0]], dtype=np.float64)


def get_head_pose(landmarks, img_w, img_h):
    """
    Estimates Pitch, Yaw, and Roll using Perspective-n-Point (PnP).
    Uses camera-specific intrinsics for accurate angle estimation.
    """
    try:
        image_points = np.array([
            landmarks[1],    # Nose tip
            landmarks[152],  # Chin
            landmarks[33],   # Left eye left corner
            landmarks[263],  # Right eye right corner
            landmarks[61],   # Left mouth corner
            landmarks[291]   # Right mouth corner
        ], dtype=np.float64)
    except IndexError:
        return 0.0, 0.0, 0.0

    cam_matrix  = _build_camera_matrix(img_w, img_h)
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, _ = cv2.solvePnP(
        FACE_3D_MODEL_POINTS,
        image_points,
        cam_matrix,
        dist_matrix,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rmat, _ = cv2.Rodrigues(rot_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    pitch = angles[0]
    yaw   = angles[1]
    roll  = angles[2]

    return pitch, yaw, roll


def calculate_aspect_ratio(landmarks, indices):
    """
    Calculates the aspect ratio using Euclidean distances between
    vertical and horizontal landmark points.
    """
    p2_p6 = np.linalg.norm(landmarks[indices[1]] - landmarks[indices[5]])
    p3_p5 = np.linalg.norm(landmarks[indices[2]] - landmarks[indices[4]])
    p1_p4 = np.linalg.norm(landmarks[indices[0]] - landmarks[indices[3]])

    if p1_p4 == 0:
        return 0.0

    return (p2_p6 + p3_p5) / (2.0 * p1_p4)


# Standard MediaPipe FaceMesh indices for Eyes and Mouth
LEFT_EYE_IDX  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33,  160, 158, 133, 153, 144]
MOUTH_IDX     = [78,  81,  13,  311, 308, 402, 14, 178]
