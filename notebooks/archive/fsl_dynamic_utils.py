# fsl_dynamic_utils.py

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
    if len(seq) < length:
        pad = [np.zeros(len(seq[0]))] * (length - len(seq))
        return seq + pad
    return seq[:length]

