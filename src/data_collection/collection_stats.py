import os
import glob
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class CollectionStats:
    def __init__(self, base_dir=None):
        """
        Provides statistics about collected sequences.
        """
        self.base_dir = Path(base_dir) if base_dir else PROJECT_ROOT / "data" / "collected"

    def get_class_counts(self):
        """
        Returns a dictionary mapping 'CATEGORY/LABEL' to the number of collected samples.
        """
        counts = {}
        if not self.base_dir.exists():
            return counts

        for category_dir in self.base_dir.iterdir():
            if not category_dir.is_dir():
                continue
                
            category_name = category_dir.name.upper().replace("_", " ")
            
            for label_dir in category_dir.iterdir():
                if not label_dir.is_dir():
                    continue
                    
                label_name = label_dir.name.upper().replace("_", " ")
                key = f"{category_name}/{label_name}"
                
                npy_files = glob.glob(str(label_dir / "seq_*.npy"))
                counts[key] = len(npy_files)
                
        return counts

    def get_stats_summary(self, target_count=80):
        """
        Returns a formatted string summary of collection progress.
        """
        counts = self.get_class_counts()
        
        if not counts:
            return "No data collected yet."
            
        summary = []
        summary.append(f"{'Category/Label':<40} | {'Count':<6} | {'Target':<6} | {'Progress'}")
        summary.append("-" * 75)
        
        for key in sorted(counts.keys()):
            count = counts[key]
            progress = min(100, int((count / target_count) * 100))
            bar = "[" + "=" * (progress // 5) + " " * (20 - (progress // 5)) + "]"
            
            summary.append(f"{key:<40} | {count:<6} | {target_count:<6} | {bar} {progress}%")
            
        return "\n".join(summary)
