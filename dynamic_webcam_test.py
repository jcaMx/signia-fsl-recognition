import cv2
import numpy as np
import pickle
from collections import deque
from tensorflow.keras.models import load_model
from src.preprocessing.normalization import normalize_landmarks
  

# Load model
model = load_model("models/dynamic/subset_lstm_model.h5", compile=False)

# Load label encoder
with open("models/dynamic/label_encoder.pkl", "rb") as f:
    le = pickle.load(f)

# Load preprocessing config
with open("models/dynamic/preprocess_config.pkl", "rb") as f:
    config = pickle.load(f)

SEQ_LENGTH = config["seq_length"]

sequence = deque(maxlen=SEQ_LENGTH)


import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera failed to open")
    exit()


from collections import deque, Counter

sequence = deque(maxlen=SEQ_LENGTH)
predictions = deque(maxlen=10)   # limit smoothing window

CONF_THRESHOLD = 0.5



print("Starting webcam...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    display_label = "..."

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y, lm.z])

        landmarks = normalize_landmarks(landmarks)
        sequence.append(landmarks)

        if len(sequence) == SEQ_LENGTH:
            input_data = np.expand_dims(sequence, axis=0)
            prediction = model.predict(input_data, verbose=0)
            print("Raw prediction:", prediction)

            confidence = np.max(prediction)
            class_id = np.argmax(prediction)

            label = le.inverse_transform([class_id])[0]
            display_label = f"{label} ({confidence:.2f})"


            if confidence > CONF_THRESHOLD:
                predictions.append(class_id)

                # smoothing
                final_class = Counter(predictions).most_common(1)[0][0]
                display_label = le.inverse_transform([final_class])[0]

            print("Confidence:", confidence)
            print("Prediction:", prediction)
            print("Sequence len:", len(sequence))
            print("Confidence:", confidence if len(sequence)==SEQ_LENGTH else "N/A")
    else:
        sequence.clear()
        predictions.clear()

    cv2.putText(frame,
                display_label,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2)

    cv2.imshow("Dynamic Sign Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()