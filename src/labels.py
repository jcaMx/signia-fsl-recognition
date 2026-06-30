import pandas as pd
from pathlib import Path

DEFAULT_LABELS_PATH = Path(__file__).resolve().parents[1] / "csv" / "expanded_labels.csv"
# Fallback to labels.csv if expanded_labels.csv doesn't exist
if not DEFAULT_LABELS_PATH.exists():
    DEFAULT_LABELS_PATH = Path(__file__).resolve().parents[1] / "csv" / "labels.csv"

def load_labels_df(csv_path=None):
    if csv_path is None:
        csv_path = DEFAULT_LABELS_PATH
    return pd.read_csv(Path(csv_path))

def get_active_labels(csv_path=None, modality=None):
    """
    Get list/dict of enabled labels, optionally filtered by modality.
    """
    df = load_labels_df(csv_path)
    # Check if 'enabled' column exists
    if 'enabled' in df.columns:
        df = df[df['enabled'].astype(str).str.lower() == 'true']
    
    if modality and 'modality' in df.columns:
        df = df[df['modality'].str.lower() == modality.lower()]
        
    return df

def get_id_to_label_map(csv_path=None, modality=None):
    df = get_active_labels(csv_path, modality)
    return dict(zip(df["id"].astype(str), df["label"]))

def get_id_to_display_map(csv_path=None, modality=None):
    df = get_active_labels(csv_path, modality)
    # Fallback to label if display_label is missing
    label_col = "display_label" if "display_label" in df.columns else "label"
    return dict(zip(df["id"].astype(str), df[label_col]))
