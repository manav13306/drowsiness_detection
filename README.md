# Real-Time Drowsiness Detection using a CNN (web app)

A CNN classifies each eye as **open** or **closed** from a webcam stream. If the eyes stay
closed longer than a set time, the page raises a visual + audible **DROWSY** alarm.
Everything runs locally: a Flask server on `localhost` and a browser UI.

## Quick start
```bash
python -m venv venv
venv\Scripts\activate            # Windows   (Linux/Mac: source venv/bin/activate)
pip install -r requirements.txt
python app.py                    # opens http://127.0.0.1:5000
```
(or double-click `run.bat` on Windows / run `./run.sh` on Linux/Mac). Click **Start camera** and
allow camera access. Python 3.9-3.12 recommended. A trained model is already included, so no
training is needed to run the app.

## What's in the UI
| Tab | Contents |
|---|---|
| **Live monitor** | Webcam with face/eye boxes, driver state, P(eyes open) meter, "eyes closed for" timer, blink count, PERCLOS (% of time eyes closed in the last minute), alert count, 60 s timeline, event log, sliders for alarm delay and open/closed threshold, sound on/off |
| **Test image / video** | Upload a photo (open/closed verdict) or a video (timeline, drowsy episodes, most-closed frame) - useful when no webcam is available |
| **Model & dataset** | Test accuracy, confusion matrix, precision/recall/F1, training curves, architecture, dataset samples |

## How it works
```
Browser webcam -> JPEG frame -> Flask /api/frame
   -> Haar face detector -> two eye boxes (from face geometry)
   -> each eye crop (26x34 gray, histogram-equalised) -> CNN -> P(open)
   -> moving average -> closed longer than threshold?  YES: DROWSY + alarm | NO: normal
```
**CNN** (`model.py` / `cnn_numpy.py`, ~290k parameters):
`[Conv3x3 -> ReLU -> MaxPool -> Dropout(0.25)] x 3 (32, 64, 128 filters) -> Dense(128) -> Dropout(0.5) -> Dense(1, sigmoid)`.
Each eye is scored at three small vertical offsets and the highest P(open) is used, so a slightly
misplaced eye box on an open eye does not trigger false alarms.

## Dataset
2,874 labelled eye crops (1,500 open / 1,374 closed), stratified split 70/15/15 (train/val/test),
see `dataset/README.md` for the source. Training uses on-the-fly augmentation (rotation, scale,
shift, flip, brightness/contrast, gamma, blur, noise). Test accuracy of the included model is
about **97-98 %** (see the *Model & dataset* tab for exact numbers).

To use a bigger dataset (e.g. MRL Eye, ~85k images):
```bash
python prepare_mrl.py --src path/to/mrlEyes_2018_01 --max-per-class 15000 --clear
python train.py
```

## Training
```bash
python train.py                  # Keras if TensorFlow is installed, otherwise the built-in NumPy trainer
python train.py --backend numpy  # force NumPy (no TensorFlow needed, ~1-2 min on a CPU)
python train.py --backend keras  # force Keras (pip install tensorflow)
```
Both trainers use the same data, split, augmentation and architecture, and both write
`models/eye_state_cnn.npz` (used by the app), `models/metrics.json`, `models/training_curves.png`
(the Keras trainer also writes `models/eye_state_cnn.keras`).

## Project structure
| File | Purpose |
|---|---|
| `app.py` | Flask server and API (`/api/frame`, `/api/analyze-image`, `/api/analyze-video`, `/api/model-info`) |
| `templates/index.html`, `static/` | Web UI (HTML, CSS, JavaScript, alarm sound) |
| `engine.py` | Detection logic: CNN classifier + drowsiness state machine (timer, blinks, PERCLOS) |
| `utils.py` | Haar face detector, eye regions, desktop alarm |
| `dataset.py` | Dataset loading, split, augmentation, preprocessing |
| `model.py` | Keras CNN definition |
| `cnn_numpy.py` | Same CNN in NumPy (inference for the app + fallback trainer) |
| `train.py` | Training and evaluation |
| `detect.py` | Optional desktop (OpenCV window) version |
| `prepare_mrl.py` | Sorts the MRL Eye dataset into `dataset/open` and `dataset/closed` |
| `tests/smoke_test.py` | End-to-end API test without a camera |
| `config.py` | All settings |

## Tuning (`config.py` or the UI sliders)
- `CLOSED_SECONDS_THRESHOLD` - seconds eyes must stay closed (a normal blink is ~0.3 s)
- `OPEN_PROB_THRESHOLD` - decision boundary of the CNN
- `SMOOTHING_FRAMES` - averaging window against flicker
- `PORT` / `HOST` - or `python app.py --port 8000 --no-browser`

## Troubleshooting
- **Camera does not start:** allow it in the browser address bar, close other apps using the camera, open the page via `http://127.0.0.1:5000` or `localhost` (browsers block cameras on other addresses).
- **"has no Haar cascades" error:** you have OpenCV 5; run `pip install "opencv-python>=4.8,<5"`.
- **Eye boxes not on the eyes:** sit upright facing the camera with the face well lit; use *Show eye boxes* to check alignment. Boxes come from the face box, so a strongly tilted head reduces accuracy.
- **Alarm silent:** keep *Alarm sound* on and make sure the tab is not muted.

## Limitations
Needs a frontal, reasonably lit face; dark sunglasses hide the eyes; an educational prototype, not a certified safety device.

## Tech
Python, OpenCV, NumPy, Flask, Keras/TensorFlow (optional), HTML/CSS/JavaScript.
