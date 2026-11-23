"""
The eye-state CNN implemented with NumPy only.

Same architecture as model.py (Keras):
    [Conv3x3 -> ReLU -> MaxPool2 -> Dropout(0.25)] x len(FILTERS) -> Flatten
    -> Dense(64, ReLU) -> Dropout(0.5) -> Dense(1, sigmoid)      output = P(eye is OPEN)

It is used (a) for inference in the web app / detect.py so TensorFlow is NOT needed to run the
project, and (b) as a fallback trainer when TensorFlow is not installed. Weights are stored in
Keras layout so the two implementations are interchangeable (see model.py).
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

import config


def _im2col(x):
    n, h, w, c = x.shape
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1), (0, 0)))
    win = sliding_window_view(xp, (3, 3), axis=(1, 2))               # N,H,W,C,3,3
    return win.transpose(0, 1, 2, 4, 5, 3).reshape(n * h * w, 9 * c)  # order (kh,kw,C) = Keras


def _col2im(dcols, shape):
    n, h, w, c = shape
    d = dcols.reshape(n, h, w, 3, 3, c)
    dxp = np.zeros((n, h + 2, w + 2, c), dcols.dtype)
    for i in range(3):
        for j in range(3):
            dxp[:, i:i + h, j:j + w, :] += d[:, :, :, i, j, :]
    return dxp[:, 1:-1, 1:-1, :]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


class NumpyCNN:
    def __init__(self, in_shape=(config.IMG_H, config.IMG_W, 1), filters=config.FILTERS,
                 dense=config.DENSE_UNITS, seed=config.SEED):
        self.in_shape, self.filters, self.dense = tuple(in_shape), tuple(filters), int(dense)
        rng = np.random.default_rng(seed)
        p, (h, w, c) = {}, in_shape
        for i, f in enumerate(self.filters):
            p[f"conv{i}_w"] = rng.normal(0, np.sqrt(2 / (9 * c)), (9 * c, f)).astype(np.float32)
            p[f"conv{i}_b"] = np.zeros(f, np.float32)
            c, h, w = f, h // 2, w // 2
        flat = h * w * c
        p["dense_w"] = rng.normal(0, np.sqrt(2 / flat), (flat, self.dense)).astype(np.float32)
        p["dense_b"] = np.zeros(self.dense, np.float32)
        p["out_w"] = rng.normal(0, np.sqrt(1 / self.dense), (self.dense, 1)).astype(np.float32)
        p["out_b"] = np.zeros(1, np.float32)
        self.p = p

    # ------------------------------------------------------------------ forward / backward
    def forward(self, x, train=False, rng=None):
        """x: float32 (N,H,W,1) in [0,1]. Returns logits (N,) and a cache for backward()."""
        cache = []
        for i in range(len(self.filters)):
            n, h, w, c = x.shape
            cols = _im2col(x)
            z = cols @ self.p[f"conv{i}_w"] + self.p[f"conv{i}_b"]
            a = np.maximum(z, 0).reshape(n, h, w, -1)
            ph, pw = h // 2, w // 2
            blocks = a[:, :ph * 2, :pw * 2, :].reshape(n, ph, 2, pw, 2, -1)
            pooled = blocks.max(axis=(2, 4))
            mask = None
            if train:
                mask = (rng.random(pooled.shape) >= 0.25).astype(np.float32) / 0.75
                out = pooled * mask
            else:
                out = pooled
            cache.append((x.shape, cols, z, a.shape, blocks, pooled, mask))
            x = out
        flat = x.reshape(len(x), -1)
        d_pre = flat @ self.p["dense_w"] + self.p["dense_b"]
        d_act = np.maximum(d_pre, 0)
        dmask = None
        if train:
            dmask = (rng.random(d_act.shape) >= 0.5).astype(np.float32) / 0.5
            d_act = d_act * dmask
        logits = (d_act @ self.p["out_w"] + self.p["out_b"]).ravel()
        return logits, (cache, x.shape, flat, d_pre, d_act, dmask)

    def loss_and_grads(self, x, y, rng):
        """Binary cross-entropy on sigmoid output; returns (loss, accuracy, grads)."""
        logits, (cache, xshape, flat, d_pre, d_act, dmask) = self.forward(x, True, rng)
        n = len(y)
        prob = _sigmoid(logits)
        eps = 1e-7
        loss = float(-np.mean(y * np.log(prob + eps) + (1 - y) * np.log(1 - prob + eps)))
        acc = float(np.mean((prob >= 0.5) == (y >= 0.5)))
        g = {}
        dz = ((prob - y) / n).astype(np.float32)[:, None]
        g["out_w"], g["out_b"] = d_act.T @ dz, dz.sum(0)
        dd = (dz @ self.p["out_w"].T) * dmask
        dd = dd * (d_pre > 0)
        g["dense_w"], g["dense_b"] = flat.T @ dd, dd.sum(0)
        dx = (dd @ self.p["dense_w"].T).reshape(xshape)
        for i in reversed(range(len(self.filters))):
            in_shape, cols, z, a_shape, blocks, pooled, mask = cache[i]
            dx = dx * mask
            n_, h, w, f = a_shape
            ph, pw = h // 2, w // 2
            sel = (blocks == pooled[:, :, None, :, None, :])
            sel = sel / sel.sum(axis=(2, 4), keepdims=True)
            da = np.zeros(a_shape, np.float32)
            da[:, :ph * 2, :pw * 2, :] = (sel * dx[:, :, None, :, None, :]).reshape(n_, ph * 2, pw * 2, f)
            dzc = da.reshape(-1, f) * (z > 0)
            g[f"conv{i}_w"], g[f"conv{i}_b"] = cols.T @ dzc, dzc.sum(0)
            if i > 0:
                dx = _col2im(dzc @ self.p[f"conv{i}_w"].T, in_shape)
        return loss, acc, g

    # ------------------------------------------------------------------ inference / io
    def predict(self, x, batch=512):
        """P(open) for float32 (N,H,W,1) in [0,1]."""
        out = [_sigmoid(self.forward(x[i:i + batch])[0]) for i in range(0, len(x), batch)]
        return np.concatenate(out) if out else np.zeros(0, np.float32)

    def n_params(self):
        return int(sum(v.size for v in self.p.values()))

    def save(self, path):
        np.savez(path, in_shape=np.array(self.in_shape), filters=np.array(self.filters),
                 dense=np.array(self.dense), **self.p)

    @classmethod
    def load(cls, path):
        d = np.load(path)
        net = cls(tuple(d["in_shape"]), tuple(int(f) for f in d["filters"]), int(d["dense"]))
        net.p = {k: d[k].astype(np.float32) for k in net.p}
        return net

    # Keras <-> NumPy weight conversion (same layer order as model.build_model)
    def to_keras_weights(self):
        c, ws = self.in_shape[2], []
        for i, f in enumerate(self.filters):
            ws += [self.p[f"conv{i}_w"].reshape(3, 3, c, f), self.p[f"conv{i}_b"]]
            c = f
        return ws + [self.p[k] for k in ("dense_w", "dense_b", "out_w", "out_b")]

    def from_keras_weights(self, ws):
        k = 0
        for i in range(len(self.filters)):
            self.p[f"conv{i}_w"] = np.asarray(ws[k], np.float32).reshape(self.p[f"conv{i}_w"].shape)
            self.p[f"conv{i}_b"] = np.asarray(ws[k + 1], np.float32)
            k += 2
        for name in ("dense_w", "dense_b", "out_w", "out_b"):
            self.p[name] = np.asarray(ws[k], np.float32).reshape(self.p[name].shape)
            k += 1
