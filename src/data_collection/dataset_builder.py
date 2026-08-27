import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class DatasetBuilder:
    def __init__(self, collected_dir=None, labels_csv=None):
        """
        Builds a combined X, y dataset from the collected .npy sequences.
        """
        self.collected_dir = Path(collected_dir) if collected_dir else PROJECT_ROOT / "data" / "collected"
        self.labels_csv = Path(labels_csv) if labels_csv else PROJECT_ROOT / "csv" / "labels.csv"
        self.label_to_id = self._load_labels()

    def _load_labels(self):
        """Loads the labels.csv to map string labels to integer IDs."""
        if not self.labels_csv.exists():
            raise FileNotFoundError(f"Labels CSV not found at {self.labels_csv}")
            
        df = pd.read_csv(self.labels_csv)
        # Create mapping of UPPERCASE label to integer ID
        mapping = dict(zip(df['label'].str.upper(), df['id']))
        return mapping

    def build_dataset(self):
        """
        Scans collected_dir for sequences and builds X and y arrays.
        Returns:
            X: np.ndarray shape (N, 30, 126)
            y: np.ndarray shape (N,)
        """
        if not self.collected_dir.exists():
            print(f"Directory {self.collected_dir} does not exist.")
            return np.empty((0, 30, 126)), np.empty((0,))

        X_list = []
        y_list = []
        
        for category_dir in self.collected_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            for label_dir in category_dir.iterdir():
                if not label_dir.is_dir():
                    continue
                    
                label_name = label_dir.name.upper().replace("_", " ")
                
                if label_name not in self.label_to_id:
                    print(f"Warning: Collected label '{label_name}' not found in labels.csv, skipping.")
                    continue
                    
                label_id = self.label_to_id[label_name]
                npy_files = glob.glob(str(label_dir / "seq_*.npy"))
                
                for f in npy_files:
                    try:
                        seq = np.load(f)
                        if seq.shape == (30, 126):
                            X_list.append(seq)
                            y_list.append(label_id)
                        else:
                            print(f"Warning: File {f} has invalid shape {seq.shape}, skipping.")
                    except Exception as e:
                        print(f"Error loading {f}: {e}")
                        
        if not X_list:
            return np.empty((0, 30, 126)), np.empty((0,))
            
        X = np.stack(X_list)
        y = np.array(y_list, dtype=np.int32)
        
        return X, y

    def merge_with_existing(self, existing_X, existing_y):
        """
        Merges newly collected dataset with an existing X/y numpy pair.
        Returns merged X, y
        """
        new_X, new_y = self.build_dataset()

        if len(new_X) == 0:
            print("No collected samples found — returning original dataset unchanged.")
            return existing_X, existing_y

        merged_X = np.concatenate([existing_X, new_X], axis=0)
        merged_y = np.concatenate([existing_y, new_y], axis=0)

        print(f"Merged {len(new_X)} collected samples into dataset. "
              f"Total: {len(merged_X)} samples.")

        return merged_X, merged_y

    def merge_with_bundle(self, bundle_path):
        """
        Loads an existing .pt bundle (containing 'X' and 'y') and merges
        webcam-collected samples into it.

        Use this before calling train() to augment the training dataset
        with data captured via the webcam collector.

        Parameters
        ----------
        bundle_path : str or Path
            Path to the existing .pt dataset bundle.

        Returns
        -------
        X : np.ndarray  shape (N, 30, 126)
        y : np.ndarray  shape (N,)  — original integer label IDs preserved
        """
        import torch
        bundle_path = Path(bundle_path)

        if not bundle_path.exists():
            raise FileNotFoundError(f"Bundle not found: {bundle_path}")

        bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)

        if "X" not in bundle or "y" not in bundle:
            raise ValueError("Bundle must contain 'X' and 'y' keys.")

        base_X = np.asarray(bundle["X"])
        base_y = np.asarray(bundle["y"])

        print(f"Loaded bundle: {bundle_path}")
        print(f"  Base dataset: {len(base_X)} samples, shape {base_X.shape}")

        return self.merge_with_existing(base_X, base_y)

    def print_summary(self):
        """
        Prints a summary of all collected samples grouped by label,
        useful for checking data balance before retraining.
        """
        X, y = self.build_dataset()

        if len(X) == 0:
            print("No collected samples found.")
            return

        # Reverse the label_to_id mapping for display
        id_to_label = {v: k for k, v in self.label_to_id.items()}

        ids, counts = np.unique(y, return_counts=True)
        print(f"\n{'Label':<30} | {'ID':>4} | {'Samples':>7}")
        print("-" * 50)
        for label_id, count in zip(ids, counts):
            label_name = id_to_label.get(int(label_id), f"ID={label_id}")
            print(f"{label_name:<30} | {int(label_id):>4} | {count:>7}")
        print("-" * 50)
        print(f"{'TOTAL':<30} | {'':>4} | {len(X):>7}\n")

