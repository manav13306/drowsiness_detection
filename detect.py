"""
Optional desktop version (OpenCV window instead of the browser UI).

    python detect.py                  # webcam
    python detect.py --video test.mp4
    python detect.py --threshold 1.5  # alarm after 1.5 s of closed eyes
Press q to quit.
"""
import argparse

import cv2

import config
from engine import DrowsinessMonitor, EyeClassifier
from utils import Alarm, FaceEyeDetector

COLORS = {"awake": (0, 200, 0), "closing": (0, 215, 255), "drowsy": (0, 0, 255), "no_face": (170, 170, 170)}
TEXT = {"awake": "AWAKE (eyes open)", "closing": "Eyes closed", "drowsy": "DROWSY!", "no_face": "NO FACE"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int, default=config.CAMERA_INDEX)
    ap.add_argument("--video", type=str, default=None)
    ap.add_argument("--threshold", type=float, default=config.CLOSED_SECONDS_THRESHOLD)
    args = ap.parse_args()

    monitor = DrowsinessMonitor(FaceEyeDetector(), EyeClassifier())
    alarm = Alarm()
    cap = cv2.VideoCapture(args.video if args.video else args.camera)
    if not cap.isOpened():
        raise SystemExit("Cannot open camera/video source.")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.video is None:
            frame = cv2.flip(frame, 1)
        res = monitor.process(frame, closed_seconds=args.threshold)
        H, W = frame.shape[:2]
        col = COLORS[res["state"]]
        if res["face"]:
            x, y, w, h = res["face"]
            cv2.rectangle(frame, (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H)), (255, 200, 0), 2)
        for e in res["eyes"]:
            x, y, w, h = e["box"]
            cv2.rectangle(frame, (int(x * W), int(y * H)), (int((x + w) * W), int((y + h) * H)),
                          (0, 200, 0) if e["p_open"] >= config.OPEN_PROB_THRESHOLD else (0, 0, 255), 2)
        cv2.putText(frame, TEXT[res["state"]], (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, col, 3)
        if res["p_open"] is not None:
            cv2.putText(frame, f"P(open)={res['p_open']:.2f}  closed {res['closed_for']:.1f}s", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        if res["alarm"]:
            alarm.start()
            cv2.rectangle(frame, (0, 0), (W - 1, H - 1), (0, 0, 255), 8)
        else:
            alarm.stop()
        cv2.imshow("Drowsiness Detection (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    alarm.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
