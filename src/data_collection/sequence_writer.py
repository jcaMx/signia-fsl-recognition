import os
import glob
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class SequenceWriter:
    def __init__(self, base_dir=None):
        """
        Manages saving collected sequences safely.
        """
        self.base_dir = Path(base_dir) if base_dir else PROJECT_ROOT / "data" / "collected"

    def _get_next_sequence_id(self, class_dir):
        """Finds the next available sequence ID in the directory."""
        if not class_dir.exists():
            return 1
            
        existing_files = glob.glob(str(class_dir / "seq_*.npy"))
        
        max_id = 0
        for f in existing_files:
            try:
                # Extract number from seq_XXX.npy
                basename = os.path.basename(f)
                num = int(basename.replace("seq_", "").replace(".npy", ""))
                max_id = max(max_id, num)
            except ValueError:
                continue
                
        return max_id + 1

    def save_sequence(self, sequence, category, label):
        """
        Saves a (30, 126) sequence to data/collected/<category>/<label>/seq_XXX.npy
        Never overwrites existing files.
        """
        if sequence.shape != (30, 126):
            raise ValueError(f"Invalid sequence shape. Expected (30, 126), got {sequence.shape}")

        if not np.isfinite(sequence).all():
            raise ValueError("Sequence contains NaN or Inf values.")

        category = category.lower().replace(" ", "_")
        label = label.lower().replace(" ", "_")
        
        class_dir = self.base_dir / category / label
        class_dir.mkdir(parents=True, exist_ok=True)
        
        next_id = self._get_next_sequence_id(class_dir)
        filename = f"seq_{next_id:03d}.npy"
        filepath = class_dir / filename
        
        np.save(filepath, sequence)
        
        return filepath
