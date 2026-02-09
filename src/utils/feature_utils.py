import numpy as np

def calculate_aspect_ratio(landmarks, indices):
    """
    Calculates the aspect ratio using the Euclidean distance between 
    vertical and horizontal points.
    """
    # Vertical distances
    p2_p6 = np.linalg.norm(landmarks[indices[1]] - landmarks[indices[5]])
    p3_p5 = np.linalg.norm(landmarks[indices[2]] - landmarks[indices[4]])
    
    # Horizontal distance
    p1_p4 = np.linalg.norm(landmarks[indices[0]] - landmarks[indices[3]])
    
    # EAR/MAR Formula
    aspect_ratio = (p2_p6 + p3_p5) / (2.0 * p1_p4)
    return aspect_ratio

# Standard MediaPipe FaceMesh indices for Eyes and Mouth
LEFT_EYE_IDX = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_IDX = [33, 160, 158, 133, 153, 144]
MOUTH_IDX = [78, 81, 13, 311, 308, 402, 14, 178] # Refined for Yawn detection