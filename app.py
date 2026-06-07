from flask import Flask, request, jsonify
from flask_cors import CORS

import base64
import numpy as np
import cv2
import pickle

from keras.models import load_model
import mediapipe as mp

from fsl_preprocessing import normalize_landmarks

# -------------------------------
# INIT
# -------------------------------
app = Flask(__name__)
CORS(app)

# Load model
model = load_model("models/static_fsl_model.keras")

# Load label encoder
with open("models/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load preprocessing config
with open("models/preprocess_config.pkl", "rb") as f:
    preprocess_config = pickle.load(f)

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

# -------------------------------
# ROUTE
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    print("PREDICT REQUEST RECEIVED")
    data = request.json['image']

    # Decode base64 image
    image_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if not result.multi_hand_landmarks:
        return jsonify({"prediction": None, "error": "No hand detected"})

    hand_landmarks = result.multi_hand_landmarks[0]

    # Extract landmarks
    landmarks = []
    raw_landmarks = []
    for lm in hand_landmarks.landmark:
        landmarks.extend([lm.x, lm.y, lm.z])
        raw_landmarks.append({"x": lm.x, "y": lm.y, "z": lm.z})

    # Apply preprocessing
    landmarks = normalize_landmarks(
        landmarks,
        scale_mode=preprocess_config['scale_mode']
    )

    X_input = np.array(landmarks).reshape(1, -1)

    # Predict
    pred_probs = model.predict(X_input, verbose=0)
    pred_class = np.argmax(pred_probs)
    letter = le.inverse_transform([pred_class])[0]

    return jsonify({
        "prediction": letter,
        "confidence": float(np.max(pred_probs)),
        "landmarks": raw_landmarks
    })

# -------------------------------
# RUN
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)