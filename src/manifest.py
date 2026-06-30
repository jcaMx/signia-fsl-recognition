import os
from pathlib import Path
from src.labels import get_active_labels

def scan_static_dataset(data_dir):
    """
    Scans a directory of static image files (organized by label name subfolders).
    Returns a list of dicts: [{'path': Path, 'label': str}]
    """
    data_dir = Path(data_dir)
    samples = []
    if not data_dir.exists():
        return samples
        
    for label_dir in data_dir.iterdir():
        if label_dir.is_dir():
            label = label_dir.name
            for file_path in label_dir.iterdir():
                if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    samples.append({
                        'path': file_path,
                        'label': label
                    })
    return samples

def scan_dynamic_dataset(data_dir, labels_csv=None):
    """
    Scans a directory of video sequences (organized by folder IDs).
    Returns a list of dicts: [{'path': Path, 'label_id': str, 'label': str}]
    """
    data_dir = Path(data_dir)
    samples = []
    if not data_dir.exists():
        return samples
        
    # Get active labels for dynamic modality
    active_labels_df = get_active_labels(labels_csv, modality='dynamic')
    id_to_label = dict(zip(active_labels_df["id"].astype(str), active_labels_df["label"]))
    
    for folder_id_dir in data_dir.iterdir():
        if folder_id_dir.is_dir() and folder_id_dir.name in id_to_label:
            label_id = folder_id_dir.name
            label_name = id_to_label[label_id]
            for file_path in folder_id_dir.iterdir():
                if file_path.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                    samples.append({
                        'path': file_path,
                        'label_id': label_id,
                        'label': label_name
                    })
    return samples
