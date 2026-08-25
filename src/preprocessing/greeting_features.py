import numpy as np


def normalize_hand(hand):
    """
    Normalize one hand represented as 63 values.

    The wrist landmark is moved to the origin and the hand is scaled by the
    maximum distance from the wrist.
    """

    hand = np.asarray(hand, dtype=np.float32).reshape(21, 3).copy()
    wrist = hand[0].copy()
    hand = hand - wrist

    distances = np.linalg.norm(hand, axis=1)
    scale = np.max(distances)

    if scale > 1e-6:
        hand = hand / scale

    return hand.reshape(-1)


def normalize_sequence(sequence):
    """
    Normalize a two-hand sequence with shape (frames, 126).
    """

    sequence = np.asarray(sequence, dtype=np.float32)
    normalized = np.zeros_like(sequence, dtype=np.float32)

    left = sequence[:, :63]
    right = sequence[:, 63:]

    for t in range(sequence.shape[0]):
        normalized[t, :63] = normalize_hand(left[t])
        normalized[t, 63:] = normalize_hand(right[t])

    return normalized


def calculate_motion(sequence):
    """
    Calculate frame-to-frame landmark movement.
    """

    sequence = np.asarray(sequence, dtype=np.float32)
    motion = np.zeros_like(sequence, dtype=np.float32)
    motion[1:] = sequence[1:] - sequence[:-1]
    return motion


def add_motion_features(X):
    """
    Concatenate landmark positions with frame-to-frame motion features.
    """

    output = []

    for sequence in np.asarray(X):
        motion = calculate_motion(sequence)
        output.append(np.concatenate([sequence, motion], axis=1))

    return np.array(output, dtype=np.float32)


def augment_sequence(sequence, noise_std=0.005, scale_range=(0.98, 1.02), rng=None):
    """
    Apply light augmentation by adding Gaussian noise and a global scale.
    """

    rng = rng or np.random
    augmented = np.asarray(sequence, dtype=np.float32).copy()

    noise = rng.normal(0, noise_std, augmented.shape)
    augmented = augmented + noise

    scale = rng.uniform(scale_range[0], scale_range[1])
    augmented = augmented * scale

    return augmented.astype(np.float32)
