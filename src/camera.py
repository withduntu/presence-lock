"""
Thin wrapper around OpenCV's VideoCapture. Keeps camera open/read/release
and failure detection in one place so main.py doesn't deal with cv2 directly.
"""

import cv2


class Camera:
    def __init__(self, camera_index: int):
        self.camera_index = camera_index
        self._cap = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.camera_index)
        return self._cap.isOpened()

    def read_frame(self):
        """Returns the frame (a numpy array), or None if the read failed."""
        if self._cap is None:
            return None
        ok, frame = self._cap.read()
        return frame if ok else None

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
