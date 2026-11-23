"""
Sort the MRL Eye Dataset into dataset/open and dataset/closed.

MRL file name: sXXXX_YYYYY_gender_glasses_EYESTATE_reflections_lighting_sensor.png
EYESTATE: 0 = closed, 1 = open.

Usage:
    python prepare_mrl.py --src path/to/mrlEyes_2018_01
"""
import argparse
import os
import shutil
import config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Extracted MRL dataset folder")
    ap.add_argument("--max-per-class", type=int, default=15000,
                    help="Limit images per class (keeps training fast)")
    ap.add_argument("--clear", action="store_true", help="Delete the bundled dataset first")
    args = ap.parse_args()

    out = {"closed": os.path.join(config.DATASET_DIR, "closed"),
           "open": os.path.join(config.DATASET_DIR, "open")}
    for p in out.values():
        if args.clear and os.path.isdir(p):
            shutil.rmtree(p)
        os.makedirs(p, exist_ok=True)
    counts = {"closed": 0, "open": 0}

    for root, _, files in os.walk(args.src):
        for f in files:
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            parts = os.path.splitext(f)[0].split("_")
            if len(parts) < 5 or parts[4] not in ("0", "1"):
                continue
            label = "open" if parts[4] == "1" else "closed"
            if counts[label] >= args.max_per_class:
                continue
            shutil.copy(os.path.join(root, f), os.path.join(out[label], f))
            counts[label] += 1
    print("Done:", counts)


if __name__ == "__main__":
    main()
