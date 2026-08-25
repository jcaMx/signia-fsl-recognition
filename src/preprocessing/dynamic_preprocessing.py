import numpy as np
from src.preprocessing.normalization import normalize_landmarks
from src.preprocessing.sequence_utils import pad_or_truncate_sequence

def trim_sequence(sequence):
    sequence = np.asarray(sequence)

    if sequence.size == 0:
        return sequence

    if sequence.ndim == 1:
        return sequence[sequence != 0]

    if sequence.ndim != 2:
        raise ValueError(f"Expected 1D or 2D sequence, got shape {sequence.shape}")

    active = ~np.all(sequence == 0, axis=1)

    return sequence[active] if np.any(active) else sequence[:0]

def get_active_window(sequence):
    active = np.any(sequence != 0, axis=1)

    if not np.any(active):
        return None, None

    start = np.argmax(active)  # first True

    end = len(active) - 1 - np.argmax(active[::-1])

    return start, end

def center_pad_sequence(sequence, length=30):
    seq = np.asarray(sequence)

    start, end = get_active_window(seq)

    if start is None:
        return np.zeros((length, seq.shape[1]))

    gesture = seq[start:end+1]

    output = np.zeros((length, seq.shape[1]))

    g_len = len(gesture)

    if g_len >= length:
        return gesture[:length]

    # center placement
    offset = (length - g_len) // 2

    output[offset:offset+g_len] = gesture

    return output


def preprocess_dynamic_sequence(sequence, length=30, scale_mode='bbox'):

    # 1. Trim FIRST (remove noise)
    sequence = trim_sequence(sequence)

    # 2. Center gesture in fixed window
    sequence = center_pad_sequence(sequence, length=length)

    # 3. Normalize each frame
    normalized_seq = []
    for frame in sequence:
        normalized_seq.append(
            normalize_landmarks(frame, scale_mode=scale_mode)
        )

    return np.array(normalized_seq)