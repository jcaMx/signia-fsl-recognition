import numpy as np
from src.preprocessing.normalization import normalize_landmarks
from src.preprocessing.sequence_utils import pad_or_truncate_sequence

def preprocess_dynamic_sequence(sequence, length=30, scale_mode='bbox'):
    """
    Given a list/sequence of raw landmarks, normalize each frame
    and pad/truncate the sequence to the fixed length.
    """
    normalized_seq = []
    for frame_landmarks in sequence:
        normalized_seq.append(normalize_landmarks(frame_landmarks, scale_mode=scale_mode))
    
    padded_seq = pad_or_truncate_sequence(normalized_seq, length=length)
    return np.array(padded_seq)
