"""Detection engine shared by the web app and detect.py: face -> eyes -> CNN -> drowsiness logic."""
import os
import time
from collections import deque

import cv2
import numpy as np

import config
from cnn_numpy import NumpyCNN
from dataset import preprocess_batch
from utils import FaceEyeDetector, crop_eye


class EyeClassifier:
    """Loads the trained CNN (models/eye_state_cnn.npz) and returns P(open) for gray eye crops."""

    def __init__(self, path=config.WEIGHTS_PATH):
        if not os.path.exists(path):
            if os.path.exists(config.KERAS_PATH):
                path = self._convert_keras()
            else:
                raise SystemExit("No trained model found. Run:  python train.py")
        self.net = NumpyCNN.load(path)

    @staticmethod
    def _convert_keras():
        from tensorflow import keras
        net = NumpyCNN()
        net.from_keras_weights(keras.models.load_model(config.KERAS_PATH).get_weights())
        net.save(config.WEIGHTS_PATH)
        return config.WEIGHTS_PATH

    def predict(self, crops):
        return self.net.predict(preprocess_batch(crops)) if crops else np.zeros(0)


class DrowsinessMonitor:
    """Per-session state machine: smoothing, closed-duration timer, blinks, PERCLOS, events."""

    def __init__(self, detector, classifier):
        self.detector, self.classifier = detector, classifier
        self.reset()

    def reset(self):
        self.smooth = deque(maxlen=config.SMOOTHING_FRAMES)
        self.closed_since = None
        self.history = deque()            # (t, closed) for PERCLOS
        self.blinks = 0
        self.drowsy_events = 0
        self.drowsy = False
        self.started = time.time()

    def process(self, frame, closed_seconds=config.CLOSED_SECONDS_THRESHOLD,
                open_thr=config.OPEN_PROB_THRESHOLD, now=None):
        """frame: BGR image. Returns a JSON-able dict (box coordinates are fractions of the frame)."""
        now = time.time() if now is None else now
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face = self.detector.detect_face(gray)
        res = {"face": None, "eyes": [], "p_open": None, "closed_for": 0.0, "new_event": False}

        if face is None:
            self.smooth.clear()
            self._end_closed(now)
            self.drowsy = False
            state = "no_face"
        else:
            x, y, w, h = face
            res["face"] = [x / W, y / H, w / W, h / H]
            # Each eye is scored at 3 vertical offsets and the best (most open) score is kept, so a
            # slightly misplaced box on an OPEN eye still reads open. Closed eyes stay closed
            # at every offset (eyelid / lashes only).
            crops, valid = [], []
            for b in self.detector.eye_regions(face):
                row = [crop_eye(gray, (b[0], b[1] + int(j * b[3]), b[2], b[3])) for j in config.EYE_JITTER]
                if all(c is not None for c in row):
                    crops += row; valid.append(b)
            probs = self.classifier.predict(crops)
            if len(probs):
                probs = probs.reshape(len(valid), len(config.EYE_JITTER)).max(axis=1)
            for b, p in zip(valid, probs):
                res["eyes"].append({"box": [b[0] / W, b[1] / H, b[2] / W, b[3] / H], "p_open": float(p)})
            if len(probs) == 0:
                state = "no_face"
            else:
                self.smooth.append(float(probs.mean()))
                p = float(np.mean(self.smooth))
                res["p_open"] = p
                closed = p < open_thr
                self.history.append((now, closed))
                if closed:
                    self.closed_since = self.closed_since or now
                    res["closed_for"] = now - self.closed_since
                    if res["closed_for"] >= closed_seconds:
                        state = "drowsy"
                        if not self.drowsy:
                            self.drowsy, res["new_event"] = True, True
                            self.drowsy_events += 1
                    else:
                        state = "closing"
                else:
                    self._end_closed(now)
                    self.drowsy = False
                    state = "awake"

        while self.history and now - self.history[0][0] > config.PERCLOS_WINDOW_S:
            self.history.popleft()
        perclos = (sum(c for _, c in self.history) / len(self.history)) if self.history else 0.0
        res.update(state=state, alarm=state == "drowsy", blinks=self.blinks,
                   drowsy_events=self.drowsy_events, perclos=perclos,
                   session_s=now - self.started)
        return res

    def _end_closed(self, now):
        """Eyes re-opened (or face lost): a short closure counts as a blink."""
        if self.closed_since is not None:
            if 0.05 <= now - self.closed_since <= 0.8:
                self.blinks += 1
            self.closed_since = None
