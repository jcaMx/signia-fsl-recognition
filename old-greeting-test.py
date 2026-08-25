import cv2
import numpy as np
import torch
import time

from collections import deque
import mediapipe as mp

from src.models.sign_lstm import SignLSTM
from src.preprocessing.greeting_features import (
    add_motion_features,
    normalize_sequence,
)


# ============================================================
# CONFIG
# ============================================================

MODEL_PATH = "notebooks/02_dynamic/best_greetings_lstm.pt"

SEQ_LENGTH = 30
INPUT_SIZE = 252
HIDDEN_SIZE = 128
NUM_LAYERS = 2
NUM_CLASSES = 5


# ============================================================
# LABEL MAPPING
# ============================================================

GREETING_CLASSES = {
    0: 'Good Morning',
    1: 'Good Afternoon',
    2: 'Good Evening',
    3: 'Hello',
    4: 'How Are You',
}

print("Classes:")
for i, label in GREETING_CLASSES.items():
    print(f"{i}: {label}")


# ============================================================
# CREATE MODEL
# ============================================================

model = SignLSTM(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES
)


# ============================================================
# LOAD TRAINED WEIGHTS
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

model.load_state_dict(checkpoint)
model.eval()

print("\nModel loaded successfully.")


# ============================================================
# MEDIAPIPE
# ============================================================

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# ============================================================
# WEBCAM
# ============================================================

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera failed to open.")
    exit()


sequence = deque(maxlen=SEQ_LENGTH)
last_predicted_label = "Waiting..."
display_until = 0

print("\nStarting webcam...")
print("Perform a greeting sign.")
print("Press Q to quit.\n")


# ============================================================
# MAIN LOOP
# ============================================================

while cap.isOpened():

    ret, frame = cap.read()

    if not ret:
        break

    image_rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    results = hands.process(image_rgb)

    now = time.time()
    display_label = last_predicted_label if now < display_until else "Waiting..."

    # --------------------------------------------------------
    # HAND DETECTED
    # --------------------------------------------------------

    if results.multi_hand_landmarks:
        print(
            "Detected hands:",
            [
                h.classification[0].label
                for h in results.multi_handedness
            ]
        )

        print("HAND DETECTED")

        # 126 features:
        # 63 left hand + 63 right hand

        left_hand = np.zeros(63, dtype=np.float32)
        right_hand = np.zeros(63, dtype=np.float32)

        if results.multi_hand_landmarks:

            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness
            ):

                landmarks = []

                for lm in hand_landmarks.landmark:
                    landmarks.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

                landmarks = np.array(
                    landmarks,
                    dtype=np.float32
                )

                # MediaPipe handedness
                hand_label = handedness.classification[0].label

                # Webcam is mirrored, so swap MediaPipe labels
                if hand_label == "Left":
                    left_hand = landmarks

                elif hand_label == "Right":
                    right_hand = landmarks

        # Combine into 126 features
        landmarks = np.concatenate([
            left_hand,
            right_hand
        ])

        print("Frame feature shape:", landmarks.shape)

        sequence.append(landmarks)
        print("Raw landmark shape:", len(landmarks))

        print("Sequence length:", len(sequence))

        if len(sequence) == SEQ_LENGTH:

            try:

                # ============================================
                # RAW SEQUENCE
                # ============================================

                raw_sequence = np.array(
                    sequence,
                    dtype=np.float32
                )

                print("\nRaw sequence shape:", raw_sequence.shape)

                # Expected:
                # (30, 126)


                # ============================================
                # NORMALIZATION
                # ============================================

                normalized = normalize_sequence(
                    raw_sequence
                )

                print(
                    "Normalized shape:",
                    normalized.shape
                )

                # Expected:
                # (30, 126)


                # ============================================
                # MOTION FEATURES
                # ============================================

                # add_motion_features expects:
                # (batch, frames, features)

                features = add_motion_features(
                    normalized[np.newaxis, ...]
                )

                print(
                    "Features with batch:",
                    features.shape
                )

                # Expected:
                # (1, 30, 252)


                # Remove batch dimension

                features = features[0]

                print(
                    "Final feature shape:",
                    features.shape
                )

                # Expected:
                # (30, 252)


                # ============================================
                # MODEL INPUT
                # ============================================

                input_tensor = torch.tensor(
                    features,
                    dtype=torch.float32
                ).unsqueeze(0)

                print(
                    "Input tensor shape:",
                    input_tensor.shape
                )

                # Expected:
                # torch.Size([1, 30, 252])


                # ============================================
                # PREDICTION
                # ============================================

                with torch.no_grad():

                    outputs = model(
                        input_tensor
                    )

                    probabilities = torch.softmax(
                        outputs,
                        dim=1
                    )

                    confidence, predicted_class = torch.max(
                        probabilities,
                        dim=1
                    )

                class_id = predicted_class.item()
                confidence = confidence.item()

                label = GREETING_CLASSES[class_id]

                last_predicted_label = f"Last predicted: {label} ({confidence:.2f})"
                display_label = last_predicted_label
                display_until = time.time() + 3

                print("\nPrediction probabilities:")

                for i, probability in enumerate(
                    probabilities[0]
                ):

                    print(
                        f"{GREETING_CLASSES[i]:20s}: "
                        f"{probability.item():.4f}"
                    )

                print(
                    f"\nPredicted: {label}"
                )

                print(
                    f"Confidence: {confidence:.4f}"
                )


            except Exception as e:

                print("\n========== ERROR ==========")
                print(type(e).__name__)
                print(str(e))
                print("============================")

                import traceback
                traceback.print_exc()

                sequence.clear()

            # --------------------------------------------------------
            # NO HAND DETECTED
            # --------------------------------------------------------

            else:

                sequence.clear()

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    cv2.putText(
        frame,
        display_label,
        (10, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "FSL Greeting Recognition",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()
hands.close()
cv2.destroyAllWindows()
