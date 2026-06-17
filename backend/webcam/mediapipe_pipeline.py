from __future__ import annotations

from threading import Lock

import cv2


class MediaPipeHandsPipeline:
    def __init__(
        self,
        static_image_mode: bool = True,
        max_num_hands: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self._lock = Lock()
        self._hands = None
        self._hands_options = {
            "static_image_mode": static_image_mode,
            "max_num_hands": max_num_hands,
            "min_detection_confidence": min_detection_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        }

    def _get_hands(self):
        if self._hands is None:
            import mediapipe as mp

            self._hands = mp.solutions.hands.Hands(**self._hands_options)
        return self._hands

    def process(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False

        with self._lock:
            return self._get_hands().process(rgb_frame)

    def close(self) -> None:
        with self._lock:
            if self._hands is not None:
                self._hands.close()
                self._hands = None
