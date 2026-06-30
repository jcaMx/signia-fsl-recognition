from src.preprocessing.normalization import normalize_landmarks

def preprocess_static_landmarks(landmarks, scale_mode='bbox'):
    """
    Apply normalization to static landmarks.
    """
    return normalize_landmarks(landmarks, scale_mode=scale_mode)
