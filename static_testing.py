from keras.models import load_model
import pickle

model = load_model("models/static_fsl_model.keras")

import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # webcam = dynamic
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

import cv2
cap = cv2.VideoCapture(0)  # 0 = default camera


import numpy as np
import pickle
from fsl_preprocessing import normalize_landmarks, preprocess_sample, extract_keypoints_from_hand_landmarks


# -------------------------------
# 1. Load artifacts
# -------------------------------

# Label encoder
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Preprocessing config
with open("models/preprocess_config.pkl", "rb") as f:
    preprocess_config = pickle.load(f)

# Example: preprocess_config might look like
# {'scale_mode': 'bbox'} or {'scale_mode': 'max_dist'}

# -------------------------------
# 2. Start webcam
# -------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Mirror view if needed (optional)
    # frame = cv2.flip(frame, 1)
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)
    
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract raw landmarks
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            
            # -------------------------------
            # 3. Apply SAME preprocessing as training
            # -------------------------------
            landmarks = normalize_landmarks(landmarks, scale_mode=preprocess_config['scale_mode'])
            
            # Convert to numpy array and reshape
            X_input = np.array(landmarks).reshape(1, -1)
            
            # Optional: check min/max
            print("Post-preprocess min/max:", X_input.min(), X_input.max())
            
            # -------------------------------
            # 4. Predict
            # -------------------------------
            pred_probs = model.predict(X_input, verbose=0)
            pred_class = np.argmax(pred_probs)
            letter = le.inverse_transform([pred_class])[0]
            
            # Display
            cv2.putText(frame, f'Prediction: {letter}', (10, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("FSL Webcam Test", frame)
    
    if cv2.waitKey(1) & 0xFF == 27:  # press Esc to exit
        break

cap.release()
cv2.destroyAllWindows()

# fsl39\Scripts\activate
# python static_testing.py