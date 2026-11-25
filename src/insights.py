"""Dataset statistics used by the Visualizations page."""
import random
import numpy as np
from PIL import Image
from . import config as C


def _count(root):
    if not root.exists():
        return {}
    return {d.name: sum(1 for f in d.iterdir() if f.is_file()) for d in sorted(root.iterdir()) if d.is_dir()}


def compute_insights(per_class_sample=60, seed=C.SEED):
    rng = random.Random(seed)
    out = {"class_counts": {"train": _count(C.TRAIN_DIR), "test": _count(C.TEST_DIR),
                            "uploads": _count(C.UPLOAD_DIR)},
           "brightness": {}, "rgb_mean": {}, "size": {}}
    for cls_dir in sorted(p for p in C.TRAIN_DIR.iterdir() if p.is_dir()):
        files = [f for f in cls_dir.iterdir() if f.is_file()]
        rng.shuffle(files)
        bright, rgbs, sizes = [], [], []
        for f in files[:per_class_sample]:
            try:
                img = Image.open(f)
                sizes.append(list(img.size))
                arr = np.asarray(img.convert("RGB").resize((64, 64)), dtype="float32")
                bright.append(float(arr.mean()))
                rgbs.append(arr.mean(axis=(0, 1)))
            except Exception:
                continue
        out["brightness"][cls_dir.name] = bright
        out["rgb_mean"][cls_dir.name] = np.mean(rgbs, axis=0).round(2).tolist() if rgbs else [0, 0, 0]
        out["size"][cls_dir.name] = sizes
    return out
