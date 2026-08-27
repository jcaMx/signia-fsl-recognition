import cv2
import numpy as np
from src.landmarks.dynamic_extractor import DynamicLandmarkExtractor
from src.preprocessing.dynamic_preprocessing import preprocess_dynamic_sequence
from src.data_collection.sequence_writer import SequenceWriter

class SequenceCollector:
    def __init__(self, sequence_length=30):
        """
        Collects sequences from webcam and processes them to standard shape (30, 126).
        """
        self.sequence_length = sequence_length
        self.extractor = DynamicLandmarkExtractor()
        self.writer = SequenceWriter()
        self.frames_buffer = []

    def start_recording(self):
        """Resets the recording buffer."""
        self.frames_buffer = []

    def process_frame(self, frame):
        """
        Extracts landmarks from a single frame and adds to buffer if hand is detected.
        Returns the current length of the buffer.
        """
        features, detected = self.extractor.extract_from_frame(frame)
        if detected:
            self.frames_buffer.append(features)
        
        return len(self.frames_buffer)

    def finalize_sequence(self, category, label):
        """
        Pads/truncates the sequence to sequence_length and normalizes it.
        Then validates and saves using SequenceWriter.
        Returns the path to the saved file if successful, or raises ValueError.
        """
        if not self.frames_buffer:
            raise ValueError("No frames collected.")

        sequence = np.array(self.frames_buffer, dtype=np.float32)
        
        # Preprocess: trim, center/pad, and normalize
        processed_sequence = preprocess_dynamic_sequence(sequence, length=self.sequence_length)
        
        # Validate
        if processed_sequence.shape != (self.sequence_length, 126):
            raise ValueError(f"Failed to generate (30, 126) sequence. Got {processed_sequence.shape}")
            
        if not np.isfinite(processed_sequence).all():
            raise ValueError("Processed sequence contains NaN or Inf.")

        # Save
        filepath = self.writer.save_sequence(processed_sequence, category, label)
        
        # Reset buffer on success
        self.start_recording()
        
        return filepath

    def close(self):
        """Releases the extractor resources."""
        self.extractor.close()
        
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
