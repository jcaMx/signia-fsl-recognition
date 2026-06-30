import cv2
import numpy as np
import pickle
from collections import deque
from tensorflow.keras.models import load_model
from src.preprocessing.normalization import normalize_landmarks
from src.inference.stabilizer import PredictionStabilizer

class DynamicInferencePipeline:
    def __init__(self, model_path, label_encoder_path, seq_length=30, scale_mode='bbox', window_size=10, confidence_threshold=0.5):
        self.model = load_model(model_path, compile=False)
        
        with open(label_encoder_path, 'rb') as f:
            self.le = pickle.load(f)
            
        self.seq_length = seq_length
        self.scale_mode = scale_mode
        self.sequence = deque(maxlen=seq_length)
        
        self.stabilizer = PredictionStabilizer(window_size=window_size, confidence_threshold=confidence_threshold)
        
        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils

    def predict_frame(self, frame_bgr):
        """
        Accumulates frame landmarks, predicts if sequence is complete,
        and returns (stable_label, raw_label, confidence, drawn_frame)
        """
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        
        drawn_frame = frame_bgr.copy()
        raw_label = None
        stable_label = None
        confidence = 0.0
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(drawn_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
            
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
                
            normalized = normalize_landmarks(landmarks, scale_mode=self.scale_mode)
            self.sequence.append(normalized)
            
            if len(self.sequence) == self.seq_length:
                input_data = np.expand_dims(self.sequence, axis=0)
                prediction = self.model.predict(input_data, verbose=0)
                
                confidence = float(np.max(prediction))
                class_id = np.argmax(prediction)
                raw_label = self.le.inverse_transform([class_id])[0]
                
                stable_class_id = self.stabilizer.add_prediction(class_id, confidence)
                if stable_class_id is not None:
                    stable_label = self.le.inverse_transform([stable_class_id])[0]
        else:
            self.sequence.clear()
            self.stabilizer.clear()
            
        return stable_label, raw_label, confidence, drawn_frame

    def close(self):
        self.hands.close()
