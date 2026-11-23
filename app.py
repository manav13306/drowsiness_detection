"""
Drowsiness Detection - web app (runs on localhost).

    python app.py                 # then open http://127.0.0.1:5000  (opens automatically)
    python app.py --port 8000 --no-browser

The browser captures the webcam (getUserMedia works on localhost) and posts JPEG frames to
/api/frame; the server runs face detection + the CNN and answers with JSON that the page draws.
"""
import argparse
import base64
import json
import os
import tempfile
import threading
import time
import webbrowser

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

import config
from engine import DrowsinessMonitor, EyeClassifier
from utils import FaceEyeDetector

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024

detector = classifier = None
sessions, sessions_lock = {}, threading.Lock()


def init_engine():
    global detector, classifier
    classifier = EyeClassifier()
    detector = FaceEyeDetector()


def get_monitor(sid):
    now = time.time()
    with sessions_lock:
        for k in [k for k, (_, t) in sessions.items() if now - t > 900]:   # drop idle sessions
            del sessions[k]
        mon = sessions[sid][0] if sid in sessions else DrowsinessMonitor(detector, classifier)
        sessions[sid] = (mon, now)
        return mon


def _float(name, default, lo, hi):
    try:
        return min(max(float(request.args.get(name, default)), lo), hi)
    except ValueError:
        return default


def decode(data):
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode the image.")
    return img


def annotate(frame, res):
    """Draw face / eye boxes and a status banner on a copy of the frame."""
    out, (H, W) = frame.copy(), frame.shape[:2]
    colors = {"awake": (80, 200, 60), "closing": (0, 190, 255), "drowsy": (60, 60, 255), "no_face": (170, 170, 170)}
    col = colors.get(res["state"], (255, 255, 255))
    if res["face"]:
        x, y, w, h = res["face"]
        cv2.rectangle(out, (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H)), (255, 200, 0), 2)
    for e in res["eyes"]:
        x, y, w, h = e["box"]
        c = (80, 200, 60) if e["p_open"] >= 0.5 else (60, 60, 255)
        cv2.rectangle(out, (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H)), c, 2)
        cv2.putText(out, f"{e['p_open']:.2f}", (int(x * W), max(int(y * H) - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
    label = {"awake": "EYES OPEN", "closing": "EYES CLOSED", "drowsy": "DROWSY", "no_face": "NO FACE"}[res["state"]]
    cv2.putText(out, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
    return out


def to_b64(img, q=80):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


# ------------------------------------------------------------------ pages / API
@app.route("/")
def index():
    return render_template("index.html", defaults={
        "closed_seconds": config.CLOSED_SECONDS_THRESHOLD, "open_thr": config.OPEN_PROB_THRESHOLD})


@app.route("/api/frame", methods=["POST"])
def api_frame():
    """Live webcam frame (raw JPEG body) -> detection result."""
    try:
        frame = decode(request.get_data())
    except ValueError as e:
        return jsonify(error=str(e)), 400
    mon = get_monitor(request.args.get("sid", "default"))
    t0 = time.time()
    res = mon.process(frame, _float("closed_s", config.CLOSED_SECONDS_THRESHOLD, 0.3, 10),
                      _float("thr", config.OPEN_PROB_THRESHOLD, 0.05, 0.95))
    res["latency_ms"] = round((time.time() - t0) * 1000, 1)
    return jsonify(res)


@app.route("/api/reset", methods=["POST"])
def api_reset():
    get_monitor(request.args.get("sid", "default")).reset()
    return jsonify(ok=True)


@app.route("/api/analyze-image", methods=["POST"])
def api_image():
    f = request.files.get("file")
    if not f:
        return jsonify(error="No file uploaded."), 400
    try:
        frame = decode(f.read())
    except ValueError as e:
        return jsonify(error=str(e)), 400
    mon = DrowsinessMonitor(detector, classifier)
    res = mon.process(frame, closed_seconds=0, open_thr=_float("thr", config.OPEN_PROB_THRESHOLD, 0.05, 0.95))
    res["verdict"] = {"awake": "Eyes open", "closing": "Eyes closed", "drowsy": "Eyes closed",
                      "no_face": "No face found"}[res["state"]]
    res["image"] = to_b64(annotate(frame, res))
    return jsonify(res)


@app.route("/api/analyze-video", methods=["POST"])
def api_video():
    f = request.files.get("file")
    if not f:
        return jsonify(error="No file uploaded."), 400
    closed_s = _float("closed_s", config.CLOSED_SECONDS_THRESHOLD, 0.3, 10)
    thr = _float("thr", config.OPEN_PROB_THRESHOLD, 0.05, 0.95)
    suffix = os.path.splitext(f.filename or "")[1] or ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        f.save(tmp.name)
    cap = cv2.VideoCapture(tmp.name)
    try:
        if not cap.isOpened():
            return jsonify(error="Could not open this video file."), 400
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, int(round(fps / 8)))            # analyse ~8 frames per second
        mon = DrowsinessMonitor(detector, classifier)
        timeline, episodes, worst, idx, cur = [], [], None, 0, None
        while idx < fps * 600:                          # at most 10 minutes of video
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                t = idx / fps
                r = mon.process(frame, closed_s, thr, now=t)
                timeline.append([round(t, 2), None if r["p_open"] is None else round(r["p_open"], 3), r["state"]])
                if r["state"] == "drowsy" and cur is None:
                    cur = t - r["closed_for"]
                if r["state"] != "drowsy" and cur is not None:
                    episodes.append([round(cur, 2), round(t, 2)]); cur = None
                if r["p_open"] is not None and (worst is None or r["p_open"] < worst[0]):
                    worst = (r["p_open"], annotate(frame, r))
            idx += 1
        if cur is not None and timeline:
            episodes.append([round(cur, 2), timeline[-1][0]])
        n_face = sum(1 for t in timeline if t[1] is not None)
        n_closed = sum(1 for t in timeline if t[1] is not None and t[1] < thr)
        return jsonify(duration=round(idx / fps, 1), analysed=len(timeline), with_face=n_face,
                       closed_pct=round(100 * n_closed / max(n_face, 1), 1), blinks=mon.blinks,
                       drowsy_events=len(episodes), episodes=episodes, timeline=timeline,
                       closed_threshold=thr,
                       frame=to_b64(worst[1]) if worst else None)
    finally:
        cap.release()
        os.unlink(tmp.name)


@app.route("/api/model-info")
def api_model_info():
    if not os.path.exists(config.METRICS_PATH):
        return jsonify(error="metrics.json not found - run python train.py"), 404
    with open(config.METRICS_PATH) as f:
        return jsonify(json.load(f))


@app.route("/models/<path:name>")
def model_files(name):
    from flask import send_from_directory
    return send_from_directory(config.MODELS_DIR, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default=config.HOST)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--no-browser", action="store_true")
    args = ap.parse_args()
    init_engine()
    url = f"http://{args.host}:{args.port}"
    print(f"\n  Drowsiness Detection running at {url}   (Ctrl+C to stop)\n")
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
