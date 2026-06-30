import cv2
import numpy as np
import mediapipe as mp
from pathlib import Path

class DynamicLandmarkExtractor:
    def __init__(self, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

    def extract_from_video_path(self, video_path):
        """
        Reads a video file and returns a list of frame landmarks (each is 63 floats).
        If hand is missing in a frame, we omit it (or it can be padded later).
        """
        cap = cv2.VideoCapture(str(video_path))
        landmark_sequence = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract landmarks for this frame
            frame_landmarks = self.extract_from_frame(frame)
            if frame_landmarks is not None:
                landmark_sequence.append(frame_landmarks)
                
        cap.release()
        return landmark_sequence

    def extract_from_frame(self, frame_bgr):
        """
        Extracts 63 floats landmarks from a single BGR frame.
        """
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        result = self.hands.process(image_rgb)
        image_rgb.flags.writeable = True
        
        if result.multi_hand_landmarks:
            # Take the first hand
            hand_landmarks = result.multi_hand_landmarks[0]
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return landmarks
        return None

    def close(self):
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
