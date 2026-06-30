import numpy as np

SEQ_LENGTH = 30  # default sequence length

def pad_or_truncate_sequence(seq, length=SEQ_LENGTH):
    """
    Pads or truncates a sequence of frame landmark vectors to a fixed length.

    Args:
        seq (list of list/np.array): sequence of frame landmarks
        length (int): desired sequence length

    Returns:
        list: sequence of length `length` (padded with zeros if necessary)
    """
    if len(seq) == 0:
        # Return sequence of zeros of size (length, 63)
        return [np.zeros(63).tolist() for _ in range(length)]
        
    if len(seq) < length:
        # seq[0] could be a numpy array or list
        feature_dim = len(seq[0])
        pad = [np.zeros(feature_dim).tolist()] * (length - len(seq))
        # Ensure elements are lists or consistent types
        seq_list = [list(x) if isinstance(x, (np.ndarray, list)) else x for x in seq]
        return seq_list + pad
    return seq[:length]
