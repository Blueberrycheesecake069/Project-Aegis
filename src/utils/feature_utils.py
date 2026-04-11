import cv2
import numpy as np

# ---------------------------------------------------------------------
# 1. 3D FACE MODEL (Generic Human Face)
# ---------------------------------------------------------------------
# Used for solving PnP to estimate head rotation.
# Points: Nose tip, Chin, Left eye corner, Right eye corner, Left mouth corner, Right mouth corner
FACE_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),          # Nose tip
    (0.0, -330.0, -65.0),     # Chin
    (-225.0, 170.0, -135.0),  # Left eye left corner
    (225.0, 170.0, -135.0),   # Right eye right corner
    (-150.0, -150.0, -125.0), # Left Mouth corner
    (150.0, -150.0, -125.0)   # Right mouth corner
], dtype=np.float64)

def get_head_pose(landmarks, img_w, img_h):
    """
    Estimates Pitch, Yaw, and Roll using Perspective-n-Point (PnP).
    """
    try:
        image_points = np.array([
            landmarks[1],   # Nose tip
            landmarks[152], # Chin
            landmarks[33],  # Left eye left corner
            landmarks[263], # Right eye right corner
            landmarks[61],  # Left mouth corner
            landmarks[291]  # Right mouth corner
        ], dtype=np.float64)
    except IndexError:
        return 0.0, 0.0, 0.0

    focal_length = 1 * img_w
    cam_matrix = np.array([ [focal_length, 0, img_h / 2],
                            [0, focal_length, img_w / 2],
                            [0, 0, 1]])
    dist_matrix = np.zeros((4, 1), dtype=np.float64)

    success, rot_vec, trans_vec = cv2.solvePnP(
        FACE_3D_MODEL_POINTS, 
        image_points, 
        cam_matrix, 
        dist_matrix, 
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rmat, jac = cv2.Rodrigues(rot_vec)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    # BUG FIXED: OpenCV already returns degrees! 
    pitch = angles[0]
    yaw   = angles[1]
    roll  = angles[2]

    return pitch, yaw, roll

def calculate_aspect_ratio(landmarks, indices):
    """
    Calculates the aspect ratio using the Euclidean distance between 
    vertical and horizontal points.
    """
    # Vertical distances
    # indices[1] top, indices[5] bottom
    p2_p6 = np.linalg.norm(landmarks[indices[1]] - landmarks[indices[5]])
    # indices[2] top, indices[4] bottom
    p3_p5 = np.linalg.norm(landmarks[indices[2]] - landmarks[indices[4]])
    
    # Horizontal distance
    # indices[0] left corner, indices[3] right corner
    p1_p4 = np.linalg.norm(landmarks[indices[0]] - landmarks[indices[3]])
    
    # EAR/MAR Formula
    # Safety check to prevent division by zero
    if p1_p4 == 0:
        return 0.0
        
    aspect_ratio = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return aspect_ratio

# Standard MediaPipe FaceMesh indices for Eyes and Mouth
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
MOUTH_IDX = [78, 81, 13, 311, 308, 402, 14, 178] # Refined for Yawn detection