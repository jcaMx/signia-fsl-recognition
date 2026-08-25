from collections import deque
import time

import cv2
import numpy as np


class WebcamRecognizer:
    def __init__(
        self,
        model,
        labels,
        seq_length=30,
        input_size=252,
        normalize_fn=None,
        feature_fn=None,
        camera_index=0,
        confidence_threshold=0.0,
        window_name="FSL Recognition",
        max_missed_frames=4,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.model = model
        self.labels = labels
        self.seq_length = seq_length
        self.input_size = input_size
        self.normalize_fn = normalize_fn
        self.feature_fn = feature_fn
        self.camera_index = camera_index
        self.confidence_threshold = confidence_threshold
        self.window_name = window_name
        self.max_missed_frames = max_missed_frames
        self.missed_frames = 0
        self.sequence = deque(maxlen=seq_length)
        self.last_label = "Waiting..."
        self.display_until = 0
        self._label_lookup = self._build_label_lookup(labels)

        import mediapipe as mp

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise RuntimeError("Camera failed to open.")

        self._logged_shape = False

    def _build_label_lookup(self, labels):
        if isinstance(labels, dict):
            return {int(k): str(v) for k, v in labels.items()}

        return {index: str(label) for index, label in enumerate(labels)}

    def _extract_landmarks(self, results):
        left_hand = np.zeros(63, dtype=np.float32)
        right_hand = np.zeros(63, dtype=np.float32)

        for hand_landmarks, handedness in zip(
            results.multi_hand_landmarks,
            results.multi_handedness,
        ):
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            landmarks = np.asarray(landmarks, dtype=np.float32)
            hand_label = handedness.classification[0].label

            # Preserve existing mirrored-webcam handedness behavior.
            if hand_label == "Left":
                left_hand = landmarks
            elif hand_label == "Right":
                right_hand = landmarks

        return np.concatenate([left_hand, right_hand]).astype(np.float32)

    def _prepare_features(self, sequence):
        raw_sequence = np.asarray(sequence, dtype=np.float32)

        if raw_sequence.ndim != 2:
            raise ValueError(
                f"Expected sequence shape (frames, features), got {raw_sequence.shape}"
            )

        if self.normalize_fn is not None:
            normalized = self.normalize_fn(raw_sequence)
        else:
            normalized = raw_sequence

        normalized = np.asarray(normalized, dtype=np.float32)

        if self.feature_fn is not None:
            features = self.feature_fn(normalized[np.newaxis, ...])
            features = np.asarray(features, dtype=np.float32)
            if features.ndim == 3:
                features = features[0]
        else:
            features = normalized

        if features.ndim != 2:
            raise ValueError(
                f"Expected final features shape (frames, features), got {features.shape}"
            )

        if features.shape[-1] != self.input_size:
            raise ValueError(
                f"Final feature dimension mismatch: expected {self.input_size}, got {features.shape[-1]}"
            )

        return features

    def _predict(self):
        raw_sequence = np.asarray(self.sequence, dtype=np.float32)
        features = self._prepare_features(raw_sequence)

        if not self._logged_shape:
            print(f"Raw sequence shape: {raw_sequence.shape}")
            print(f"Final feature shape: {features.shape}")
            print(f"Model input shape: {(1,) + features.shape}")
            self._logged_shape = True

        input_tensor = np.expand_dims(features, axis=0)
        try:
            import torch

            input_tensor = torch.tensor(input_tensor, dtype=torch.float32)
            device = next(self.model.parameters()).device
            input_tensor = input_tensor.to(device)

            self.model.eval()
            with torch.no_grad():
                outputs = self.model(input_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted_class = torch.max(probabilities, dim=1)

            confidence = float(confidence.item())
            class_id = int(predicted_class.item())

        except Exception as exc:
            raise RuntimeError(f"Prediction failed: {exc}") from exc

        if class_id not in self._label_lookup:
            raise ValueError(
                f"Predicted class id {class_id} not found in labels."
            )

        label = self._label_lookup[class_id]
        if confidence < self.confidence_threshold:
            label = "Waiting..."

        return label, confidence, probabilities[0].detach().cpu().numpy()

    def run(self):
        print("\nStarting webcam...")
        print("Press Q to quit.\n")

        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)

            now = time.time()
            display_label = self.last_label if now < self.display_until else "Waiting..."

            if results.multi_hand_landmarks:
                self.missed_frames = 0

                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                    )

                landmarks = self._extract_landmarks(results)
                self.sequence.append(landmarks)

                if len(self.sequence) == self.seq_length:
                    try:
                        label, confidence, probabilities = self._predict()
                        if label != "Waiting...":
                            self.last_label = f"Last predicted: {label} ({confidence:.2f})"
                            display_label = self.last_label
                            self.display_until = time.time() + 3

                            print("\nPrediction probabilities:")
                            for i, probability in enumerate(probabilities):
                                label_name = self._label_lookup.get(i, str(i))
                                print(f"{label_name:20s}: {probability:.4f}")
                            print(f"\nPredicted: {label}")
                            print(f"Confidence: {confidence:.4f}")
                    except Exception as exc:
                        print("\n========== ERROR ==========")
                        print(type(exc).__name__)
                        print(str(exc))
                        print("============================")
                        import traceback
                        traceback.print_exc()
                        self.sequence.clear()
            else:
                self.missed_frames += 1
                if self.missed_frames > self.max_missed_frames:
                    self.sequence.clear()
                    self.last_label = "Waiting..."
                    self.display_until = 0

            cv2.putText(
                frame,
                display_label,
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            cv2.imshow(self.window_name, frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        self.close()

    def close(self):
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        if hasattr(self, "hands") and self.hands is not None:
            self.hands.close()
        cv2.destroyAllWindows()
