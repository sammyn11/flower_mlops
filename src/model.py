"""Model creation, training, evaluation and RETRAINING.

Initial model : MobileNetV2 (ImageNet) transfer learning + dropout + L2 +
                EarlyStopping + ReduceLROnPlateau + two-phase fine-tuning.
Retraining    : loads OUR OWN saved model (models/flower_model.h5) as the
                pre-trained base, then fine-tunes on uploaded data + a replay
                sample of the original data (prevents catastrophic forgetting).

CLI:  python -m src.model train | retrain | evaluate
"""
import json
import random
import shutil
import sys
from datetime import datetime

import numpy as np

from . import config as C
from . import database as db
from . import preprocessing as P


# ------------------------------------------------------------------ build
def build_model(num_classes, dropout=0.3, l2=1e-4):
    import tensorflow as tf
    from tensorflow.keras import layers, regularizers
    base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                             input_shape=(*C.IMG_SIZE, 3))
    base.trainable = False
    inputs = tf.keras.Input(shape=(*C.IMG_SIZE, 3))
    x = layers.Rescaling(1 / 127.5, offset=-1)(inputs)        # [0,255] -> [-1,1]
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation="softmax",
                           kernel_regularizer=regularizers.l2(l2))(x)
    return tf.keras.Model(inputs, outputs, name="flower_classifier")


def compile_model(model, lr):
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(lr),
                  loss="categorical_crossentropy", metrics=["accuracy"])


def unfreeze_top(model, n_layers=30):
    """Unfreeze the last n layers of the MobileNetV2 base (BatchNorm stays frozen)."""
    import tensorflow as tf
    base = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers[-n_layers:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)


def make_callbacks(progress_cb=None, patience=3):
    import tensorflow as tf
    cbs = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience,
                                            restore_best_weights=True),
           tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7)]
    if progress_cb:
        cbs.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda e, logs: progress_cb(
            "training", f"epoch {e + 1}: loss={logs['loss']:.4f} acc={logs['accuracy']:.4f} "
                        f"val_loss={logs['val_loss']:.4f} val_acc={logs['val_accuracy']:.4f}")))
    return cbs


# ------------------------------------------------------------------ evaluate
def evaluate_model(model, class_names, directory=C.TEST_DIR):
    """Returns (metrics_dict, y_true, y_pred). Metrics: accuracy, loss, precision, recall, F1."""
    import tensorflow as tf
    from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                                 classification_report)
    paths, labels = P.list_files(directory, class_names)
    ds = P.paths_to_dataset(paths, labels, len(class_names), training=False)
    probs = model.predict(ds, verbose=0)
    y_true, y_pred = np.array(labels), probs.argmax(axis=1)
    loss = float(tf.keras.losses.CategoricalCrossentropy()(tf.one_hot(y_true, len(class_names)), probs).numpy())
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred)), "loss": loss,
               "precision": float(p), "recall": float(r), "f1": float(f1),
               "n_test": int(len(y_true)),
               "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
               "per_class": classification_report(y_true, y_pred, target_names=class_names,
                                                  output_dict=True, zero_division=0)}
    return metrics, y_true, y_pred


# ------------------------------------------------------------------ persistence
def load_meta():
    if C.META_PATH.exists():
        return json.loads(C.META_PATH.read_text())
    return {"version": 0, "history": []}


def save_artifacts(model, class_names, metrics, source, extra=None):
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    meta = load_meta()
    version = meta.get("version", 0) + 1
    model.save(C.MODEL_PATH)
    archive = C.MODELS_DIR / "archive"
    archive.mkdir(exist_ok=True)
    shutil.copy(C.MODEL_PATH, archive / f"flower_model_v{version}.h5")
    C.CLASSES_PATH.write_text(json.dumps(class_names))
    entry = {"version": version, "source": source, "trained_at": datetime.utcnow().isoformat(),
             "metrics": metrics, **(extra or {})}
    meta.update({"version": version, "class_names": class_names, "latest": entry})
    meta.setdefault("history", []).append({k: entry[k] for k in ("version", "source", "trained_at")}
                                          | {"accuracy": metrics["accuracy"], "f1": metrics["f1"]})
    C.META_PATH.write_text(json.dumps(meta, indent=2))
    return version


# ------------------------------------------------------------------ initial training
def train_initial(head_epochs=8, ft_epochs=8, progress_cb=None):
    import tensorflow as tf
    log = progress_cb or (lambda s, m: print(f"[{s}] {m}"))
    tf.keras.utils.set_random_seed(C.SEED)
    class_names = P.get_class_names(C.TRAIN_DIR)
    log("preprocessing", f"classes={class_names}")
    paths, labels = P.list_files(C.TRAIN_DIR, class_names)
    (tp, tl), (vp, vl) = P.stratified_split(paths, labels, 0.15)
    train_ds = P.paths_to_dataset(tp, tl, len(class_names), training=True)
    val_ds = P.paths_to_dataset(vp, vl, len(class_names))
    log("preprocessing", f"{len(tp)} train / {len(vp)} validation images")

    model = build_model(len(class_names))
    compile_model(model, 1e-3)
    log("training", "phase 1: training classification head (base frozen)")
    model.fit(train_ds, validation_data=val_ds, epochs=head_epochs, callbacks=make_callbacks(log), verbose=0)
    log("training", "phase 2: fine-tuning top MobileNetV2 layers at low LR")
    unfreeze_top(model, 30)
    compile_model(model, 1e-5)
    model.fit(train_ds, validation_data=val_ds, epochs=ft_epochs, callbacks=make_callbacks(log), verbose=0)

    log("evaluating", "evaluating on held-out test set")
    metrics, _, _ = evaluate_model(model, class_names)
    version = save_artifacts(model, class_names, metrics, "initial_training")
    log("done", f"saved model v{version}: acc={metrics['accuracy']:.4f} f1={metrics['f1']:.4f}")
    return load_meta()


# ------------------------------------------------------------------ RETRAINING
def retrain(progress_cb=None, epochs=6, replay_per_class=40, lr=1e-5):
    """Fine-tune the existing saved model on uploaded data (+ replay of old data)."""
    import tensorflow as tf
    log = progress_cb or (lambda s, m: print(f"[{s}] {m}"))
    if not C.MODEL_PATH.exists():
        raise FileNotFoundError("No trained model found. Train an initial model first.")
    class_names = json.loads(C.CLASSES_PATH.read_text())

    log("loading", f"loading existing model {C.MODEL_PATH.name} as the pre-trained base")
    model = tf.keras.models.load_model(C.MODEL_PATH)
    before_metrics, _, _ = evaluate_model(model, class_names)
    log("loading", f"current model test accuracy = {before_metrics['accuracy']:.4f}")

    new_paths, new_labels = P.list_files(C.UPLOAD_DIR, class_names)
    if not new_paths:
        raise ValueError("No uploaded data found. Upload images first.")
    log("preprocessing", f"{len(new_paths)} uploaded images (already validated, resized to "
                         f"{C.IMG_SIZE[0]}x{C.IMG_SIZE[1]} and stored at upload time)")

    old_paths, old_labels = P.list_files(C.TRAIN_DIR, class_names)
    rng = random.Random(C.SEED)
    replay_p, replay_l = [], []
    for y in range(len(class_names)):
        idx = [i for i, l in enumerate(old_labels) if l == y]
        rng.shuffle(idx)
        for i in idx[:replay_per_class]:
            replay_p.append(old_paths[i]); replay_l.append(old_labels[i])
    paths, labels = new_paths + replay_p, new_labels + replay_l
    (tp, tl), (vp, vl) = P.stratified_split(paths, labels, 0.15)
    train_ds = P.paths_to_dataset(tp, tl, len(class_names), training=True)
    val_ds = P.paths_to_dataset(vp, vl, len(class_names))
    log("preprocessing", f"combined set: {len(new_paths)} new + {len(replay_p)} replay -> "
                         f"{len(tp)} train / {len(vp)} validation")

    unfreeze_top(model, 30)
    compile_model(model, lr)
    log("training", f"fine-tuning for up to {epochs} epochs (lr={lr})")
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=make_callbacks(log, patience=2), verbose=0)

    log("evaluating", "evaluating retrained model on held-out test set")
    after_metrics, _, _ = evaluate_model(model, class_names)
    version = save_artifacts(model, class_names, after_metrics, "retraining",
                             {"new_images_used": len(new_paths),
                              "accuracy_before": before_metrics["accuracy"]})
    db.mark_used(version)
    log("done", f"saved model v{version}: accuracy {before_metrics['accuracy']:.4f} -> "
                f"{after_metrics['accuracy']:.4f}")
    return load_meta()


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "train"
    if cmd == "train":
        train_initial()
    elif cmd == "retrain":
        retrain()
    elif cmd == "evaluate":
        import tensorflow as tf
        names = json.loads(C.CLASSES_PATH.read_text())
        m, _, _ = evaluate_model(tf.keras.models.load_model(C.MODEL_PATH), names)
        print(json.dumps({k: v for k, v in m.items() if k not in ("confusion_matrix", "per_class")}, indent=2))
    else:
        print("usage: python -m src.model [train|retrain|evaluate]")
