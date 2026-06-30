import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from pathlib import Path

class StaticLandmarkExtractor:
    def __init__(self, model_asset_path="hand_landmarker.task"):
        # Resolve path relative to project root if it exists
        proj_root = Path(__file__).resolve().parents[2]
        resolved_path = proj_root / model_asset_path
        if not resolved_path.exists():
            resolved_path = Path(model_asset_path)
            
        self.options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(resolved_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_hands=1
        )
        self.landmarker = vision.HandLandmarker.create_from_options(self.options)

    def extract_from_image_path(self, img_path):
        """
        Loads image, runs hand landmark detection, and returns flattened list of 63 floats.
        """
        image = cv2.imread(str(img_path))
        if image is None:
            return None
        return self.extract_from_frame(image)

    def extract_from_frame(self, frame_bgr):
        """
        Runs detection on an already loaded BGR frame.
        """
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=image_rgb
        )
        result = self.landmarker.detect(mp_image)
        
        if result.hand_landmarks:
            landmarks = []
            for lm in result.hand_landmarks[0]:
                landmarks.extend([lm.x, lm.y, lm.z])
            return landmarks
        return None

    def close(self):
        self.landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
