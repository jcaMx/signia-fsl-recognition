import cv2
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from src.landmarks.static_extractor import StaticLandmarkExtractor
from src.preprocessing.static_preprocessing import preprocess_static_landmarks

class StaticInferencePipeline:
    def __init__(self, model_path, label_encoder_path, scale_mode='bbox'):
        self.model = load_model(model_path)
        
        with open(label_encoder_path, 'rb') as f:
            self.le = pickle.load(f)
            
        self.scale_mode = scale_mode
        # Use MediaPipe Solutions Hands for live webcam video tracking efficiency
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
        Processes a single frame and returns (prediction_label, confidence, drawn_frame)
        """
        rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb_frame)
        
        label = None
        confidence = 0.0
        drawn_frame = frame_bgr.copy()
        
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(drawn_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Extract raw landmarks
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y, lm.z])
                
                # Preprocess
                normalized_landmarks = preprocess_static_landmarks(landmarks, scale_mode=self.scale_mode)
                X_input = np.array(normalized_landmarks).reshape(1, -1)
                
                # Predict
                pred_probs = self.model.predict(X_input, verbose=0)
                pred_class = np.argmax(pred_probs)
                confidence = float(np.max(pred_probs))
                label = self.le.inverse_transform([pred_class])[0]
                
        return label, confidence, drawn_frame

    def close(self):
        self.hands.close()
