"""Central configuration for the Drowsiness Detection project."""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Paths ----
DATASET_DIR = os.path.join(BASE_DIR, "dataset")            # dataset/open , dataset/closed
MODELS_DIR = os.path.join(BASE_DIR, "models")
WEIGHTS_PATH = os.path.join(MODELS_DIR, "eye_state_cnn.npz")     # used by the app (no TensorFlow needed)
KERAS_PATH = os.path.join(MODELS_DIR, "eye_state_cnn.keras")     # written only when trained with Keras
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
ALARM_PATH = os.path.join(BASE_DIR, "static", "alarm.wav")

# ---- CNN / training ----
IMG_H, IMG_W = 26, 34       # eye crop size (same as the bundled dataset)
FILTERS = (32, 64, 128)     # conv blocks
DENSE_UNITS = 128
BATCH_SIZE = 64
EPOCHS = 40
LEARNING_RATE = 1e-3
SEED = 42
VAL_FRAC, TEST_FRAC = 0.15, 0.15

# ---- Detection ----
CAMERA_INDEX = 0
OPEN_PROB_THRESHOLD = 0.5        # P(open) below this => eyes closed
CLOSED_SECONDS_THRESHOLD = 1.5   # eyes closed for this long => DROWSY
NO_FACE_SECONDS_THRESHOLD = 3.0
EYE_JITTER = (-0.12, 0.0, 0.12)  # vertical crop offsets (fraction of box height) tried per eye
SMOOTHING_FRAMES = 3             # moving average over last N predictions
PERCLOS_WINDOW_S = 60            # window for "percentage of eye closure"

# ---- Web app ----
HOST = "127.0.0.1"
PORT = 5000
