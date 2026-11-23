"""
End-to-end smoke test of the web API (no camera needed).

    python tests/smoke_test.py

Uses the real Haar face detector when OpenCV 4.x is installed. On OpenCV builds without Haar
cascades it falls back to a fixed face box for the bundled sample portrait, so the CNN + logic
+ API are still exercised.
"""
import io, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, numpy as np

import app as webapp
import config
from dataset import load_dataset
from engine import EyeClassifier
from utils import FaceEyeDetector

SAMPLE_FACE = (172, 60, 104, 110)          # face box of the sample portrait (x, y, w, h)


def sample_portrait():
    try:
        from skimage import data
        return cv2.cvtColor(data.astronaut(), cv2.COLOR_RGB2BGR)
    except ImportError:
        sys.exit("pip install scikit-image to run the portrait-based smoke test")


class FixedDetector(FaceEyeDetector):
    def __init__(self): pass
    def detect_face(self, gray):
        return SAMPLE_FACE if gray.mean() > 40 else None


def closed_version(img, det, X, y):
    """Paste real closed-eye crops (held-out test split, clearly closed) over both eye regions."""
    from dataset import preprocess_batch, stratified_split
    out = img.copy(); rng = np.random.default_rng(1)
    te = stratified_split(y)[2]; te = te[y[te] == 0]
    sure = te[webapp.classifier.net.predict(preprocess_batch(X[te])) < 0.2]
    for b in det.eye_regions(SAMPLE_FACE):
        crop = X[rng.choice(sure)]
        x0, y0, w, h = b
        out[y0:y0 + h, x0:x0 + w] = cv2.cvtColor(cv2.resize(crop, (w, h)), cv2.COLOR_GRAY2BGR)
    return out


def jpg(img): return cv2.imencode(".jpg", img)[1].tobytes()


def main():
    real = hasattr(cv2, "CascadeClassifier")
    webapp.classifier = EyeClassifier()
    webapp.detector = FaceEyeDetector() if real else FixedDetector()
    det = webapp.detector
    print("face detector:", "Haar cascade" if real else "fixed-box stub (no Haar in this OpenCV)")
    face = SAMPLE_FACE if not real else None
    X, y = load_dataset()
    open_img = sample_portrait()
    if real:
        face = det.detect_face(cv2.cvtColor(open_img, cv2.COLOR_BGR2GRAY)); assert face is not None, "no face found"
        globals()["SAMPLE_FACE"] = face
    closed_img = closed_version(open_img, det, X, y)
    c = webapp.app.test_client()

    r = c.get("/"); assert r.status_code == 200 and b"Drowsiness" in r.data; print("index page ok")
    for p in ("/static/app.js", "/static/style.css", "/static/alarm.wav", "/api/model-info", "/models/dataset_samples.png"):
        assert c.get(p).status_code == 200, p
    print("static files + model-info ok")

    def frame(img, **kw):
        q = "&".join(f"{k}={v}" for k, v in dict(sid="t", closed_s=0.4, thr=0.5, **kw).items())
        r = c.post(f"/api/frame?{q}", data=jpg(img), content_type="image/jpeg"); assert r.status_code == 200
        return r.get_json()

    c.post("/api/reset?sid=t")
    for _ in range(4): res = frame(open_img)
    print("open eyes   ->", res["state"], f"P(open)={res['p_open']:.2f}"); assert res["state"] == "awake"
    res = frame(np.zeros_like(open_img)); print("blank frame ->", res["state"]); assert res["state"] == "no_face"
    states = []
    for _ in range(8):
        res = frame(closed_img); states.append(res["state"]); time.sleep(0.12)
    print("closed eyes ->", states, f"P(open)={res['p_open']:.2f}")
    assert states[0] == "closing" and states[-1] == "drowsy" and res["alarm"] and res["drowsy_events"] == 1
    res = frame(open_img); res = frame(open_img); res = frame(open_img)
    print("eyes reopen ->", res["state"]); assert res["state"] == "awake" and not res["alarm"]

    r = c.post("/api/analyze-image", data={"file": (io.BytesIO(jpg(closed_img)), "a.jpg")}); j = r.get_json()
    print("image upload ->", j["verdict"], j["image"][:22]); assert j["verdict"] == "Eyes closed"
    r = c.post("/api/analyze-image", data={"file": (io.BytesIO(b"not an image"), "a.jpg")}); assert r.status_code == 400

    path = "/tmp/_smoke.avi"
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 10, (open_img.shape[1], open_img.shape[0]))
    for i in range(70): vw.write(open_img if i < 20 else closed_img if i < 55 else open_img)
    vw.release()
    with open(path, "rb") as f:
        r = c.post("/api/analyze-video?closed_s=1.0", data={"file": (f, "t.avi")}); j = r.get_json()
    print("video upload ->", {k: j[k] for k in ("duration", "analysed", "closed_pct", "drowsy_events", "episodes")})
    assert j["drowsy_events"] == 1 and 3.0 <= j["closed_pct"] / 100 * j["duration"] <= 4.5
    print("\nALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
