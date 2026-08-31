"""
video_processor.py
------------------
Processes pre-recorded video files to extract FSL sign language sequences.
Standardizes the output to (30, 126) for dataset compatibility or inference.
Designed for integration with a backend (e.g., Flask) where users upload videos.
"""

import numpy as np
import logging
from pathlib import Path
from typing import Union

from src.landmarks.dynamic_extractor import DynamicLandmarkExtractor
from src.preprocessing.dynamic_preprocessing import preprocess_dynamic_sequence
from src.data_collection.sequence_writer import SequenceWriter

logger = logging.getLogger(__name__)

class VideoProcessor:
    def __init__(self, sequence_length: int = 30):
        """
        Processes video files into normalized landmark sequences.
        """
        self.sequence_length = sequence_length
        self.extractor = DynamicLandmarkExtractor()
        self.writer = SequenceWriter()

    def process_to_array(self, video_path: Union[str, Path]) -> np.ndarray:
        """
        Extracts landmarks from the video and normalizes to (30, 126).
        Useful for future inference where we just want the array without saving.

        Returns
        -------
        np.ndarray
            Shape (30, 126) containing the processed landmarks.

        Raises
        ------
        ValueError
            If no hands were detected in the video or processing failed.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        logger.info(f"Extracting landmarks from video: {video_path}")
        raw_sequence, metadata = self.extractor.extract_video(video_path)

        if len(raw_sequence) == 0:
            raise ValueError(f"No hands detected in the video: {video_path}")
            
        logger.debug(f"Extraction metadata: {metadata}")

        # Preprocess: trim, center/pad, and normalize
        processed_sequence = preprocess_dynamic_sequence(raw_sequence, length=self.sequence_length)

        # Validate
        if processed_sequence.shape != (self.sequence_length, 126):
            raise ValueError(
                f"Failed to generate ({self.sequence_length}, 126) sequence. "
                f"Got {processed_sequence.shape}"
            )

        if not np.isfinite(processed_sequence).all():
            raise ValueError("Processed sequence contains NaN or Inf.")

        return processed_sequence

    def process_and_save(self, video_path: Union[str, Path], category: str, label: str) -> Path:
        """
        Extracts landmarks from the video, normalizes to 30 frames, 
        and saves it to the standard dataset structure as an .npy file.
        
        Returns
        -------
        Path
            The path to the newly saved .npy file.
        """
        processed_sequence = self.process_to_array(video_path)
        
        filepath = self.writer.save_sequence(processed_sequence, category, label)
        logger.info(f"Saved processed video sequence to {filepath}")
        
        return filepath

    def close(self):
        """Releases the extractor resources."""
        self.extractor.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
