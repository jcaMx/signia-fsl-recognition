import numpy as np

def normalize_landmarks(landmarks, scale_mode='bbox'):
    """
    Normalize hand landmarks relative to wrist and scale them.

    Args:
        landmarks: list or array of 63 floats (21 points x 3)
        scale_mode: 'bbox', 'max_dist', or 'none'

    Returns:
        normalized_landmarks: list of 63 floats normalized
    """
    landmarks = np.array(landmarks).reshape(21, 3)
    wrist = landmarks[0]

    # Make wrist origin
    landmarks -= wrist

    if scale_mode == 'bbox':
        # Scale to bounding box [-1, 1]
        min_vals = landmarks.min(axis=0)
        max_vals = landmarks.max(axis=0)
        scale = max(max_vals - min_vals)
        if scale > 0:
            landmarks /= scale
    elif scale_mode == 'max_dist':
        # Scale by max distance to wrist
        dists = np.linalg.norm(landmarks, axis=1)
        max_dist = dists.max()
        if max_dist > 0:
            landmarks /= max_dist
    # else: 'none' -> leave as is

    return landmarks.flatten().tolist()
