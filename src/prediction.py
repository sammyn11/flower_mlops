"""Model loading + single-image prediction. Auto-reloads if the model file changes
(important when several API containers share one models/ volume)."""
import json
import threading

import numpy as np

from . import config as C
from . import preprocessing as P


class Predictor:
    def __init__(self):
        self._model = None
        self._mtime = None
        self.class_names = []
        self._lock = threading.Lock()
        self._predict_lock = threading.Lock()

    def _load_if_needed(self):
        if not C.MODEL_PATH.exists():
            raise FileNotFoundError("Model not found. Run `python -m src.model train` first.")
        mtime = C.MODEL_PATH.stat().st_mtime
        if self._model is None or mtime != self._mtime:
            import tensorflow as tf
            with self._lock:
                self._model = tf.keras.models.load_model(C.MODEL_PATH)
                self.class_names = json.loads(C.CLASSES_PATH.read_text())
                self._mtime = mtime

    @property
    def loaded(self):
        return self._model is not None

    def reload(self):
        self._model = None
        self._load_if_needed()

    def predict(self, image_bytes: bytes):
        self._load_if_needed()
        x = P.preprocess_bytes(image_bytes)
        # Keras predict() is not safe/effective when many request threads invoke
        # it on the same model at once. Serialize inference per API replica;
        # separate replicas still serve predictions in parallel.
        with self._predict_lock:
            probs = self._model.predict(x, verbose=0)[0]
        i = int(np.argmax(probs))
        return {"label": self.class_names[i], "confidence": float(probs[i]),
                "probabilities": {c: float(p) for c, p in zip(self.class_names, probs)}}


predictor = Predictor()


def predict_image(image_bytes: bytes):
    return predictor.predict(image_bytes)
