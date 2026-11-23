"""Helpers: face/eye-region extraction and desktop alarm playback."""
import os
import sys
import threading
import time

import cv2
import numpy as np

import config


class FaceEyeDetector:
    """Haar-cascade face detector + eye regions derived from the face box.

    Eye cascades often miss closed eyes, so the two eye boxes are computed from the face
    geometry. That keeps a crop available exactly when the CNN needs to see a closed eye.
    """

    def __init__(self):
        if not hasattr(cv2, "CascadeClassifier"):
            raise RuntimeError("This OpenCV build has no Haar cascades. "
                               "Install a 4.x build:  pip install \"opencv-python>=4.8,<5\"")
        path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        self.face_cascade = cv2.CascadeClassifier(path)
        if self.face_cascade.empty():
            raise RuntimeError(f"Could not load Haar face cascade from {path}")
        self._lock = threading.Lock()      # cascades are not thread-safe

    def detect_face(self, gray):
        """Largest face as (x, y, w, h) or None. Works on a downscaled copy for speed."""
        scale = 1.0
        if gray.shape[1] > 480:
            scale = 480.0 / gray.shape[1]
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        min_side = max(40, int(0.12 * gray.shape[1]))
        with self._lock:
            faces = self.face_cascade.detectMultiScale(
                cv2.equalizeHist(gray), scaleFactor=1.1, minNeighbors=5, minSize=(min_side, min_side))
        if len(faces) == 0:
            return None
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        return tuple(int(v / scale) for v in (x, y, w, h))

    @staticmethod
    def eye_regions(face):
        """(left_box, right_box) as (x, y, w, h); 'left' = left side of the image."""
        x, y, w, h = face
        ew, eh = int(0.33 * w), int(0.25 * h)
        ey = y + int(0.235 * h)
        return (x + int(0.115 * w), ey, ew, eh), (x + int(0.555 * w), ey, ew, eh)


def crop_eye(gray, box):
    x, y, w, h = box
    H, W = gray.shape[:2]
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    return gray[y0:y1, x0:x1]


class Alarm:
    """Looping alarm for the desktop script (pygame if available, else beep)."""

    def __init__(self, wav_path=config.ALARM_PATH):
        self.active = False
        self._pygame = None
        try:
            import pygame
            pygame.mixer.init()
            self._sound = pygame.mixer.Sound(wav_path)
            self._pygame = pygame
        except Exception as e:
            print(f"[alarm] pygame unavailable ({e}); using fallback beep.")

    def start(self):
        if self.active:
            return
        self.active = True
        if self._pygame:
            self._sound.play(loops=-1)
        else:
            threading.Thread(target=self._beep_loop, daemon=True).start()

    def stop(self):
        if not self.active:
            return
        self.active = False
        if self._pygame:
            self._sound.stop()

    def _beep_loop(self):
        while self.active:
            if sys.platform.startswith("win"):
                import winsound
                winsound.Beep(1800, 400)
            else:
                print("\a", end="", flush=True)
                time.sleep(0.5)
