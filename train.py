"""
Train the eye-state CNN on dataset/open and dataset/closed.

    python train.py                    # uses TensorFlow/Keras if installed, else the NumPy trainer
    python train.py --backend numpy    # force the NumPy trainer (no TensorFlow needed)
    python train.py --backend keras    # force Keras
    python train.py --epochs 60

Outputs (models/):  eye_state_cnn.npz (used by the app), eye_state_cnn.keras (Keras only),
                    metrics.json (accuracy, confusion matrix, curves), training_curves.png
"""
import argparse
import json
import os
import time

import cv2
import numpy as np

import config
from cnn_numpy import NumpyCNN
from dataset import CLASSES, augment, load_dataset, preprocess_batch, stratified_split


# ------------------------------------------------------------------ metrics
def evaluate(prob, y, thr=0.5):
    pred = (prob >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    prec_o, rec_o = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    prec_c, rec_c = tn / max(tn + fn, 1), tn / max(tn + fp, 1)
    f1 = lambda p, r: 2 * p * r / max(p + r, 1e-9)
    return {"accuracy": (tp + tn) / max(len(y), 1),
            "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},      # rows=true, cols=pred (closed,open)
            "open": {"precision": prec_o, "recall": rec_o, "f1": f1(prec_o, rec_o)},
            "closed": {"precision": prec_c, "recall": rec_c, "f1": f1(prec_c, rec_c)}}


def bce(prob, y):
    e = 1e-7
    return float(-np.mean(y * np.log(prob + e) + (1 - y) * np.log(1 - prob + e)))


# ------------------------------------------------------------------ trainers
def train_numpy(Xtr, ytr, Xva, yva, epochs, rng):
    net = NumpyCNN()
    m = {k: np.zeros_like(v) for k, v in net.p.items()}
    v = {k: np.zeros_like(x) for k, x in net.p.items()}
    t, hist, best, best_p, wait = 0, {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}, -1, None, 0
    lr, b1, b2 = config.LEARNING_RATE, 0.9, 0.999
    Xva_p = preprocess_batch(Xva)
    for ep in range(1, epochs + 1):
        t0 = time.time()
        Xa = preprocess_batch(augment(Xtr, rng))
        order = rng.permutation(len(Xa))
        L, A = [], []
        for i in range(0, len(order), config.BATCH_SIZE):
            idx = order[i:i + config.BATCH_SIZE]
            loss, acc, g = net.loss_and_grads(Xa[idx], ytr[idx].astype(np.float32), rng)
            t += 1
            for k in net.p:
                m[k] = b1 * m[k] + (1 - b1) * g[k]
                v[k] = b2 * v[k] + (1 - b2) * g[k] ** 2
                mh, vh = m[k] / (1 - b1 ** t), v[k] / (1 - b2 ** t)
                net.p[k] -= (lr * mh / (np.sqrt(vh) + 1e-8)).astype(np.float32)
            L.append(loss); A.append(acc)
        pv = net.predict(Xva_p)
        vl, va = bce(pv, yva), float(np.mean((pv >= .5) == yva))
        for k, val in zip(hist, (np.mean(L), np.mean(A), vl, va)):
            hist[k].append(float(val))
        print(f"epoch {ep:2d}/{epochs}  loss {np.mean(L):.4f} acc {np.mean(A):.4f}  "
              f"val_loss {vl:.4f} val_acc {va:.4f}  ({time.time() - t0:.1f}s)", flush=True)
        if va > best + 1e-4:
            best, best_p, wait = va, {k: x.copy() for k, x in net.p.items()}, 0
        else:
            wait += 1
            if wait == 5:
                lr *= 0.5; print(f"  lr -> {lr:.2e}")
            if wait >= 12:
                print("early stopping"); break
    net.p = best_p
    return net, hist


def train_keras(Xtr, ytr, Xva, yva, epochs, rng):
    from model import build_model
    model = build_model()
    model.summary()
    Xva_p = preprocess_batch(Xva)
    hist = {"loss": [], "accuracy": [], "val_loss": [], "val_accuracy": []}
    best, best_w, wait, lr = -1, None, 0, config.LEARNING_RATE
    for ep in range(1, epochs + 1):               # fresh augmentation every epoch
        Xa = preprocess_batch(augment(Xtr, rng))
        h = model.fit(Xa, ytr.astype("float32"), batch_size=config.BATCH_SIZE, epochs=1,
                      validation_data=(Xva_p, yva.astype("float32")), verbose=2)
        for k in hist:
            hist[k].append(float(h.history[k][0]))
        va = hist["val_accuracy"][-1]
        if va > best + 1e-4:
            best, best_w, wait = va, model.get_weights(), 0
        else:
            wait += 1
            if wait == 5:
                lr *= 0.5
                model.optimizer.learning_rate.assign(lr)
            if wait >= 12:
                print("early stopping"); break
    model.set_weights(best_w)
    net = NumpyCNN()
    net.from_keras_weights(best_w)               # same weights, NumPy layout (used by the app)
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    model.save(config.KERAS_PATH)
    return net, hist


# ------------------------------------------------------------------ outputs
def save_plots(hist, conf):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    ax[0].plot(hist["accuracy"], label="train"); ax[0].plot(hist["val_accuracy"], label="val")
    ax[0].set_title("Accuracy"); ax[0].set_xlabel("epoch"); ax[0].legend()
    ax[1].plot(hist["loss"], label="train"); ax[1].plot(hist["val_loss"], label="val")
    ax[1].set_title("Loss"); ax[1].set_xlabel("epoch"); ax[1].legend()
    cm = np.array([[conf["tn"], conf["fp"]], [conf["fn"], conf["tp"]]])
    ax[2].imshow(cm, cmap="Blues"); ax[2].set_title("Test confusion matrix")
    ax[2].set_xticks([0, 1], ["closed", "open"]); ax[2].set_yticks([0, 1], ["closed", "open"])
    ax[2].set_xlabel("predicted"); ax[2].set_ylabel("true")
    for (i, j), val in np.ndenumerate(cm):
        ax[2].text(j, i, val, ha="center", va="center",
                   color="white" if val > cm.max() / 2 else "black", fontsize=14)
    fig.tight_layout()
    fig.savefig(os.path.join(config.MODELS_DIR, "training_curves.png"), dpi=110)


def save_samples(X, y, rng):
    rows = []
    for lab in (1, 0):
        sel = rng.choice(np.where(y == lab)[0], 20, replace=False)
        rows.append(np.vstack([np.hstack([X[i] for i in sel[r * 10:(r + 1) * 10]]) for r in range(2)]))
    sheet = np.vstack([rows[0], np.full((6, rows[0].shape[1]), 255, np.uint8), rows[1]])
    cv2.imwrite(os.path.join(config.MODELS_DIR, "dataset_samples.png"),
                cv2.resize(sheet, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=("auto", "keras", "numpy"), default="auto")
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = ap.parse_args()

    backend = args.backend
    if backend == "auto":
        try:
            import tensorflow  # noqa: F401
            backend = "keras"
        except ImportError:
            backend = "numpy"
    print(f"Backend: {backend}")

    X, y = load_dataset()
    tr, va, te = stratified_split(y)
    counts = {c: int((y == i).sum()) for i, c in enumerate(CLASSES)}
    print(f"Dataset: {len(y)} eye crops {counts} | train {len(tr)}  val {len(va)}  test {len(te)}")
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    rng = np.random.default_rng(config.SEED)

    t0 = time.time()
    fn = train_keras if backend == "keras" else train_numpy
    net, hist = fn(X[tr], y[tr], X[va], y[va], args.epochs, rng)
    net.save(config.WEIGHTS_PATH)

    prob = net.predict(preprocess_batch(X[te]))
    test = evaluate(prob, y[te])
    c = test["confusion"]
    print(f"\nTEST accuracy {test['accuracy'] * 100:.2f}%   confusion {c}")
    metrics = {"backend": backend, "train_seconds": round(time.time() - t0, 1),
               "dataset": {"total": int(len(y)), **counts, "train": int(len(tr)),
                           "val": int(len(va)), "test": int(len(te)),
                           "image_size": [config.IMG_H, config.IMG_W]},
               "test": test, "history": hist, "params": net.n_params(),
               "architecture": [f"Input {config.IMG_H}x{config.IMG_W}x1"] +
                               [f"Conv3x3({f}) + ReLU + MaxPool + Dropout(0.25)" for f in config.FILTERS] +
                               ["Flatten", f"Dense({config.DENSE_UNITS}) + ReLU + Dropout(0.5)", "Dense(1) + Sigmoid"]}
    with open(config.METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)
    save_plots(hist, c)
    save_samples(X, y, rng)
    print(f"Saved {config.WEIGHTS_PATH}")


if __name__ == "__main__":
    main()
