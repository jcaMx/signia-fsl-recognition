import cv2
import numpy as np
import mediapipe as mp


class DynamicLandmarkExtractor:
    """
    Extracts dynamic hand landmark sequences from videos.

    Output shape:
        sequence -> (num_frames, 126)

    where:
        left hand  = 21 * 3 = 63
        right hand = 21 * 3 = 63
        total      = 126
    """

    FEATURE_SIZE = 126
    HAND_SIZE = 63

    def __init__(
        self,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ):
        self.mp_hands = mp.solutions.hands

        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    ####################################################################
    # Frame Extraction
    ####################################################################

    def extract_from_frame(self, frame):
        """
        Extract landmarks from a single frame.

        Returns
        -------
        features : np.ndarray
            Shape: (126,)
        detected : bool
            True if at least one hand was detected
        """

        image = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB,
        )

        result = self.hands.process(image)

        left_hand = np.zeros(
            self.HAND_SIZE,
            dtype=np.float32,
        )

        right_hand = np.zeros(
            self.HAND_SIZE,
            dtype=np.float32,
        )

        detected = False

        if result.multi_hand_landmarks:

            detected = True

            for landmarks, handedness in zip(
                result.multi_hand_landmarks,
                result.multi_handedness,
            ):

                coords = []

                for lm in landmarks.landmark:
                    coords.extend([
                        lm.x,
                        lm.y,
                        lm.z,
                    ])

                coords = np.array(
                    coords,
                    dtype=np.float32,
                )

                label = (
                    handedness
                    .classification[0]
                    .label
                )

                if label == "Left":
                    left_hand = coords
                else:
                    right_hand = coords

        features = np.concatenate([
            left_hand,
            right_hand,
        ])

        return features, detected

    ####################################################################
    # Video Extraction
    ####################################################################

    def extract_video(self, video_path):
        """
        Extract a landmark sequence from a video.

        Returns
        -------
        sequence : np.ndarray
            Shape:
                (num_frames, 126)

        metadata : dict
            {
                total_frames,
                detected_frames,
                detection_rate
            }
        """

        cap = cv2.VideoCapture(
            str(video_path)
        )

        sequence = []

        total_frames = 0
        detected_frames = 0

        while cap.isOpened():

            ret, frame = cap.read()

            if not ret:
                break

            total_frames += 1

            features, detected = (
                self.extract_from_frame(frame)
            )

            if detected:
                detected_frames += 1

            sequence.append(features)

        cap.release()

        if len(sequence) > 0:
            sequence = np.array(
                sequence,
                dtype=np.float32,
            )
        else:
            sequence = np.empty(
                (0, self.FEATURE_SIZE),
                dtype=np.float32,
            )

        metadata = {
            "total_frames": total_frames,
            "detected_frames": detected_frames,
            "detection_rate": (
                detected_frames / total_frames
                if total_frames > 0
                else 0.0
            ),
        }

        return sequence, metadata

    ####################################################################
    # Cleanup
    ####################################################################

    def close(self):
        self.hands.close()

    def __enter__(self):
        return self

    def __exit__(
        self,
        exc_type,
        exc_val,
        exc_tb,
    ):
        self.close()