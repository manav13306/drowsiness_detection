"""Dataset loading, splitting, augmentation and eye-crop preprocessing (shared by all code)."""
import os
import cv2
import numpy as np
import config

CLASSES = ("closed", "open")        # label 0 = closed, 1 = open
EXTS = (".png", ".jpg", ".jpeg", ".bmp")


def preprocess_batch(imgs):
    """uint8 (N,H,W) gray crops -> float32 (N,H,W,1) in [0,1], contrast-equalised."""
    out = np.empty((len(imgs), config.IMG_H, config.IMG_W, 1), np.float32)
    for i, im in enumerate(imgs):
        if im.shape != (config.IMG_H, config.IMG_W):
            im = cv2.resize(im, (config.IMG_W, config.IMG_H), interpolation=cv2.INTER_AREA)
        out[i, :, :, 0] = cv2.equalizeHist(im) / 255.0
    return out


def load_dataset(root=config.DATASET_DIR):
    """Return uint8 images (N,H,W) and labels (N,) read from root/closed and root/open."""
    xs, ys = [], []
    for label, name in enumerate(CLASSES):
        folder = os.path.join(root, name)
        if not os.path.isdir(folder):
            raise SystemExit(f"Dataset folder missing: {folder}  (see README.md)")
        for f in sorted(os.listdir(folder)):
            if f.lower().endswith(EXTS):
                im = cv2.imread(os.path.join(folder, f), cv2.IMREAD_GRAYSCALE)
                if im is None:
                    continue
                xs.append(cv2.resize(im, (config.IMG_W, config.IMG_H), interpolation=cv2.INTER_AREA))
                ys.append(label)
    if not xs:
        raise SystemExit("No images found in dataset/open and dataset/closed.")
    return np.array(xs, np.uint8), np.array(ys, np.int64)


def stratified_split(y, val_frac=config.VAL_FRAC, test_frac=config.TEST_FRAC, seed=config.SEED):
    """Return index arrays (train, val, test) keeping the class balance in every split."""
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        n_te, n_va = int(len(idx) * test_frac), int(len(idx) * val_frac)
        te += list(idx[:n_te]); va += list(idx[n_te:n_te + n_va]); tr += list(idx[n_te + n_va:])
    return np.array(sorted(tr)), np.array(sorted(va)), np.array(sorted(te))


def augment(imgs, rng):
    """Random rotation / scale / shift / flip / brightness / contrast / blur / noise (uint8 in & out)."""
    h, w = config.IMG_H, config.IMG_W
    out = np.empty_like(imgs)
    for i, im in enumerate(imgs):
        ang = rng.uniform(-12, 12)
        sc = rng.uniform(0.72, 1.3)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, sc)
        M[0, 2] += rng.uniform(-4.5, 4.5)
        M[1, 2] += rng.uniform(-4.5, 4.5)
        a = cv2.warpAffine(im, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        if rng.random() < 0.5:
            a = a[:, ::-1]
        a = a.astype(np.float32)
        a = a * rng.uniform(0.6, 1.4) + rng.uniform(-35, 35)          # contrast / brightness
        if rng.random() < 0.3:
            a = 255.0 * (np.clip(a, 0, 255) / 255.0) ** rng.uniform(0.6, 1.6)   # gamma
        if rng.random() < 0.3:
            a = cv2.GaussianBlur(a, (3, 3), rng.uniform(0.3, 1.0))            # webcam softness
        a = a + rng.normal(0, rng.uniform(0, 7), a.shape)                     # sensor noise
        out[i] = np.clip(a, 0, 255).astype(np.uint8)
    return out
