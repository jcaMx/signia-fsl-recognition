"""
process_video.py
----------------
CLI tool to test extracting FSL sign language sequences from a local video file.
Saves the resulting (30, 126) sequence to the collected dataset.

Usage:
    python scripts/process_video.py <video_path> <category> <label>

Example:
    python scripts/process_video.py test_video.mp4 GREETING "GOOD MORNING"
"""

import sys
import os
import argparse
import logging
from pathlib import Path

# ============================================================
# Project root
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# Logging
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

from src.data_collection.video_processor import VideoProcessor
from src.data_collection.label_registry import LabelRegistry

def main():
    parser = argparse.ArgumentParser(description="Process a video file into an FSL .npy sequence.")
    parser.add_argument("video_path", type=str, help="Path to the input video file (e.g. video.mp4)")
    parser.add_argument("category", type=str, help="Category of the sign (e.g. GREETING)")
    parser.add_argument("label", type=str, help="Label of the sign (e.g. 'GOOD MORNING')")
    
    args = parser.parse_args()
    
    video_path = Path(args.video_path)
    if not video_path.exists():
        logger.error(f"Video file not found: {video_path}")
        sys.exit(1)
        
    try:
        registry = LabelRegistry()
    except Exception as e:
        logger.error(f"Failed to load LabelRegistry: {e}")
        sys.exit(1)
        
    # Validate category and label
    category = args.category.strip().upper()
    label = args.label.strip().upper()
    
    if category not in registry.get_categories():
        logger.error(f"Category '{category}' not found in labels.csv")
        sys.exit(1)
        
    if label not in registry.get_labels(category):
        logger.error(f"Label '{label}' not found under category '{category}' in labels.csv")
        sys.exit(1)
        
    logger.info(f"Processing video: {video_path}")
    logger.info(f"Target class: {category} / {label}")
    
    try:
        with VideoProcessor() as processor:
            saved_path = processor.process_and_save(video_path, category, label)
            logger.info(f"Success! Sequence saved to: {saved_path}")
    except Exception as e:
        logger.error(f"Failed to process video: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
