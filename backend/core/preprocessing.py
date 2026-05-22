from __future__ import annotations

import numpy as np

LANDMARK_COUNT = 21
COORDINATES_PER_LANDMARK = 3
EXPECTED_FEATURE_COUNT = LANDMARK_COUNT * COORDINATES_PER_LANDMARK


def get_first_hand_landmarks(mediapipe_result):
    hand_sets = getattr(mediapipe_result, "multi_hand_landmarks", None)
    if not hand_sets:
        return None
    return hand_sets[0]


def get_hand_count(mediapipe_result) -> int:
    hand_sets = getattr(mediapipe_result, "multi_hand_landmarks", None)
    if not hand_sets:
        return 0
    return len(hand_sets)


def flatten_xyz_landmarks(hand_landmarks) -> list[float]:
    features: list[float] = []
    for landmark in hand_landmarks.landmark:
        features.extend((landmark.x, landmark.y, landmark.z))

    if len(features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} landmark features, got {len(features)}."
        )

    return features


def normalize_landmarks(features: list[float], scale_mode: str = "bbox") -> list[float]:
    landmarks = np.asarray(features, dtype=np.float32).reshape(LANDMARK_COUNT, COORDINATES_PER_LANDMARK)
    wrist = landmarks[0].copy()

    # Match training-time preprocessing by expressing all landmarks relative to the wrist.
    landmarks -= wrist

    if scale_mode == "bbox":
        min_vals = landmarks.min(axis=0)
        max_vals = landmarks.max(axis=0)
        scale = float(np.max(max_vals - min_vals))
        if scale > 0:
            landmarks /= scale
    elif scale_mode == "max_dist":
        distances = np.linalg.norm(landmarks, axis=1)
        max_distance = float(distances.max())
        if max_distance > 0:
            landmarks /= max_distance

    return landmarks.flatten().tolist()


def build_single_frame_input(
    features: list[float],
    expected_feature_count: int = EXPECTED_FEATURE_COUNT,
) -> np.ndarray:
    if len(features) != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} input features, got {len(features)}."
        )
    return np.asarray([features], dtype=np.float32)
