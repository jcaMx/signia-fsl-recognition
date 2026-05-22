from __future__ import annotations

from pathlib import Path

import numpy as np


def load_labels() -> list[str]:
    labels_path = Path(__file__).with_name("static_label_classes.npy")
    if labels_path.exists():
        return np.load(labels_path, allow_pickle=True).tolist()

    return list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


ALPHABET_LABELS = load_labels()
