import numpy as np

def _normalize_single_hand(hand, scale_mode):
    """Normalize one hand (21,3)"""
    wrist = hand[0]
    hand = hand - wrist

    if scale_mode == 'bbox':
        min_vals = hand.min(axis=0)
        max_vals = hand.max(axis=0)
        scale = np.max(max_vals - min_vals)
        if scale > 0:
            hand = hand / scale

    elif scale_mode == 'max_dist':
        dists = np.linalg.norm(hand, axis=1)
        scale = np.max(dists)
        if scale > 0:
            hand = hand / scale

    return hand


def normalize_landmarks(landmarks, scale_mode='bbox'):
    """
    Supports:
    - 63 = single hand
    - 126 = two hands (left + right)
    Returns: flattened vector (63 or 126 normalized)
    """

    landmarks = np.asarray(landmarks)

    # -------------------------
    # SINGLE HAND
    # -------------------------
    if landmarks.shape[0] == 63:
        hand = landmarks.reshape(21, 3)
        hand = _normalize_single_hand(hand, scale_mode)
        return hand.flatten().tolist()

    # -------------------------
    # TWO HANDS
    # -------------------------
    if landmarks.shape[0] == 126:
        left = landmarks[:63].reshape(21, 3)
        right = landmarks[63:].reshape(21, 3)

        left = _normalize_single_hand(left, scale_mode)
        right = _normalize_single_hand(right, scale_mode)

        # IMPORTANT: keep structure consistent
        return np.concatenate([left.flatten(), right.flatten()]).tolist()

    raise ValueError(f"Unexpected landmark size: {landmarks.shape}")