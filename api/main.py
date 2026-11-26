"""FastAPI service: predict, upload, train/retrain, status, insights."""
import json
import os
import socket
import threading
import time
from collections import deque
from datetime import datetime
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from src import config as C
from src import database as db
from src.insights import compute_insights
from src.prediction import predictor

START_TIME = time.time()
HOST = socket.gethostname()           # container id -> shows load balancing in action
recent_conf = deque(maxlen=50)

app = FastAPI(title="Flower Classifier MLOps API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------- job state (shared file)
def read_state():
    try:
        return json.loads(C.STATE_PATH.read_text())
    except Exception:
        return {"status": "idle", "kind": None, "stage": None, "logs": [], "error": None, "updated": 0}


def write_state(s):
    s["updated"] = time.time()
    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = C.STATE_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(s))
    os.replace(tmp, C.STATE_PATH)


def job_log(stage, msg):
    s = read_state()
    s["stage"] = stage
    s["logs"] = (s["logs"] + [f"{datetime.utcnow():%H:%M:%S} [{stage}] {msg}"])[-200:]
    write_state(s)


def _run_job(kind):
    from src import model as M
    try:
        if kind == "train":
            M.train_initial(progress_cb=job_log)
        else:
            M.retrain(progress_cb=job_log)
        predictor.reload()
        s = read_state(); s["status"] = "done"; write_state(s)
    except Exception as e:                                   # noqa: BLE001
        s = read_state(); s["status"] = "error"; s["error"] = str(e)
        s["logs"].append(f"ERROR: {e}"); write_state(s)


def start_job(kind):
    s = read_state()
    if s["status"] == "running" and time.time() - s["updated"] < 1800:
        return False
    write_state({"status": "running", "kind": kind, "stage": "starting", "logs": [], "error": None})
    threading.Thread(target=_run_job, args=(kind,), daemon=True).start()
    return True


# ---------------------------------------------------------------- endpoints
@app.get("/")
def root():
    return {"service": "flower-classifier", "docs": "/docs"}


@app.get("/status")
def status():
    meta = {}
    if C.META_PATH.exists():
        meta = json.loads(C.META_PATH.read_text())
    avg = sum(recent_conf) / len(recent_conf) if recent_conf else None
    return {"status": "up", "served_by": HOST,
            "uptime_seconds": round(time.time() - START_TIME, 1),
            "model_exists": C.MODEL_PATH.exists(),
            "model_version": meta.get("version"),
            "model_metrics": {k: v for k, v in meta.get("latest", {}).get("metrics", {}).items()
                              if k in ("accuracy", "loss", "precision", "recall", "f1", "n_test")},
            "pending_uploads": db.pending_count(),
            "total_uploads": db.total_count(),
            "retrain_threshold": C.RETRAIN_THRESHOLD,
            "avg_recent_confidence": avg,
            "retrain_recommended": bool(avg is not None and len(recent_conf) >= 20 and avg < C.LOW_CONFIDENCE),
            "job": {k: read_state()[k] for k in ("status", "kind", "stage")}}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    t0 = time.perf_counter()
    try:
        result = await run_in_threadpool(predictor.predict, data)
    except FileNotFoundError as e:
        raise HTTPException(503, str(e))
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(400, f"Could not process image: {e}")
    recent_conf.append(result["confidence"])
    meta = json.loads(C.META_PATH.read_text()) if C.META_PATH.exists() else {}
    return {**result, "model_version": meta.get("version"), "served_by": HOST,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 1)}


@app.post("/upload")
async def upload(label: str = Form(...), files: List[UploadFile] = File(...)):
    """Bulk upload of labelled images for retraining (preprocessed + saved + logged)."""
    from src.preprocessing import save_uploaded_image
    if not C.CLASSES_PATH.exists():
        raise HTTPException(503, "Train an initial model first.")
    classes = json.loads(C.CLASSES_PATH.read_text())
    if label not in classes:
        raise HTTPException(400, f"Unknown label '{label}'. Valid labels: {classes}")
    saved, rejected = 0, []
    for f in files:
        try:
            save_uploaded_image(await f.read(), f.filename, label)
            saved += 1
        except Exception:                                    # noqa: BLE001
            rejected.append(f.filename)
    pending = db.pending_count()
    auto = False
    if C.AUTO_RETRAIN and pending >= C.RETRAIN_THRESHOLD:
        auto = start_job("retrain")
    return {"saved": saved, "rejected": rejected, "pending_uploads": pending,
            "auto_retrain_triggered": auto, "threshold": C.RETRAIN_THRESHOLD}


@app.post("/retrain")
def retrain():
    if db.total_count() == 0:
        raise HTTPException(400, "No uploaded data. Upload images first.")
    if not start_job("retrain"):
        raise HTTPException(409, "A training job is already running.")
    return {"message": "Retraining started", "poll": "/job"}


@app.post("/train")
def train():
    if not start_job("train"):
        raise HTTPException(409, "A training job is already running.")
    return {"message": "Initial training started (from ImageNet weights)", "poll": "/job"}


@app.get("/job")
def job():
    return read_state()


@app.get("/insights")
def insights():
    cache = C.DATA_DIR / "insights.json"
    try:
        newest = max(p.stat().st_mtime for p in [C.TRAIN_DIR, C.UPLOAD_DIR] if p.exists())
        if cache.exists() and cache.stat().st_mtime >= newest:
            return json.loads(cache.read_text())
        data = compute_insights()
        cache.write_text(json.dumps(data))
        return data
    except Exception as e:                                   # noqa: BLE001
        raise HTTPException(500, str(e))


@app.get("/metrics")
def metrics():
    if not C.META_PATH.exists():
        raise HTTPException(404, "No model metadata yet.")
    return json.loads(C.META_PATH.read_text())
