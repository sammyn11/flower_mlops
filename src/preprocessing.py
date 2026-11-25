"""Data acquisition + preprocessing.

Pipeline: download -> stratified train/test split -> resize to 224x224 ->
(augment, training only) -> batch. Pixel scaling to [-1, 1] happens INSIDE the
model (Rescaling layer) so training and serving can never disagree.
"""
import io
import random
import shutil
import tarfile
import urllib.request
import uuid
from collections import defaultdict

import numpy as np
from PIL import Image

from . import config as C
from . import database as db

DATASET_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
IMG_EXT = {".jpg", ".jpeg", ".png"}


# ------------------------------------------------------------------ acquisition
def download_and_split(test_fraction=0.2, seed=C.SEED):
    """Download TF Flowers and create data/train and data/test (stratified per class)."""
    if any(C.TRAIN_DIR.glob("*/*")) and any(C.TEST_DIR.glob("*/*")):
        print("Train/test folders already populated - skipping download.")
        return
    raw = C.DATA_DIR / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / "flower_photos.tgz"
    if not archive.exists():
        print("Downloading dataset (~220 MB)...")
        urllib.request.urlretrieve(DATASET_URL, archive)
    root = raw / "flower_photos"
    if not root.exists():
        with tarfile.open(archive) as t:
            t.extractall(raw)
    rng = random.Random(seed)
    for cls_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        files = sorted(f for f in cls_dir.iterdir() if f.suffix.lower() in IMG_EXT)
        rng.shuffle(files)
        n_test = int(len(files) * test_fraction)
        for i, f in enumerate(files):
            dest = (C.TEST_DIR if i < n_test else C.TRAIN_DIR) / cls_dir.name
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest / f.name)
    print("Split complete.")


def get_class_names(directory=C.TRAIN_DIR):
    return sorted(p.name for p in directory.iterdir() if p.is_dir())


def list_files(root, class_names):
    paths, labels = [], []
    for idx, cls in enumerate(class_names):
        d = root / cls
        if not d.exists():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMG_EXT:
                paths.append(str(f))
                labels.append(idx)
    return paths, labels


def stratified_split(paths, labels, val_fraction=0.15, seed=C.SEED):
    by_class = defaultdict(list)
    for p, y in zip(paths, labels):
        by_class[y].append(p)
    rng = random.Random(seed)
    tr, va = ([], []), ([], [])
    for y, ps in by_class.items():
        rng.shuffle(ps)
        n_val = max(1, int(len(ps) * val_fraction))
        for p in ps[:n_val]:
            va[0].append(p); va[1].append(y)
        for p in ps[n_val:]:
            tr[0].append(p); tr[1].append(y)
    return tr, va


# ------------------------------------------------------------------ tf.data pipeline
def get_augmenter():
    import tensorflow as tf
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.1),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ], name="augmentation")


def paths_to_dataset(paths, labels, num_classes, training=False, batch_size=C.BATCH_SIZE):
    import tensorflow as tf
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=C.SEED, reshuffle_each_iteration=True)

    def _load(path, y):
        img = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        img = tf.image.resize(img, C.IMG_SIZE)
        img.set_shape((*C.IMG_SIZE, 3))
        return img, tf.one_hot(y, num_classes)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)
    if training:
        aug = get_augmenter()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ------------------------------------------------------------------ serving / uploads
def preprocess_bytes(data: bytes):
    """Raw image bytes -> (1, 224, 224, 3) float32 array (0-255; model rescales)."""
    img = Image.open(io.BytesIO(data)).convert("RGB").resize(C.IMG_SIZE, Image.BILINEAR)
    return np.expand_dims(np.asarray(img, dtype="float32"), 0)


def save_uploaded_image(data: bytes, filename: str, label: str):
    """Preprocess an uploaded training image and store it + a DB record.

    Steps: validate it is a real image -> convert to RGB -> resize to 224x224 ->
    save as JPEG in data/uploads/<label>/ -> log in SQLite.
    """
    img = Image.open(io.BytesIO(data))
    img.verify()                                  # raises if corrupt / not an image
    img = Image.open(io.BytesIO(data)).convert("RGB").resize(C.IMG_SIZE, Image.BILINEAR)
    dest_dir = C.UPLOAD_DIR / label
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{uuid.uuid4().hex}.jpg"
    img.save(dest, "JPEG", quality=92)
    db.add_upload(filename, label, dest)
    return dest
