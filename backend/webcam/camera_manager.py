from __future__ import annotations

import cv2


class CameraManager:
    def __init__(
        self,
        camera_index: int = 0,
        api_preference: int = getattr(cv2, 'CAP_DSHOW', 0),
        frame_width: int = 640,
        frame_height: int = 480,
        warmup_frames: int = 3,
    ) -> None:
        self.camera_index = camera_index
        self.api_preference = api_preference
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.warmup_frames = warmup_frames
        self._camera = None
        self._is_warmed_up = False

    def _open_camera(self) -> None:
        if self._camera is None or not self._camera.isOpened():
            self._camera = cv2.VideoCapture(self.camera_index, self.api_preference)
            if self._camera is not None and self._camera.isOpened():
                self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
                self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
                self._camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self._is_warmed_up = False

    def _warmup_camera(self) -> None:
        if self._camera is None or not self._camera.isOpened() or self._is_warmed_up:
            return

        for _ in range(self.warmup_frames):
            self._camera.read()
        self._is_warmed_up = True

    def read_frame(self):
        self._open_camera()
        if self._camera is None or not self._camera.isOpened():
            return False, None
        self._warmup_camera()

        success = False
        frame = None
        for _ in range(2):
            success, frame = self._camera.read()
            if success and frame is not None:
                break
        return success, frame

    def release(self) -> None:
        if self._camera is not None and self._camera.isOpened():
            self._camera.release()
        self._is_warmed_up = False
