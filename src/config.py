"""Central configuration: paths and constants shared by every module."""
import os
from pathlib import Path

ROOT = Path(os.getenv("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
DATA_DIR = ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
UPLOAD_DIR = DATA_DIR / "uploads"          # new data uploaded for retraining
DB_PATH = DATA_DIR / "uploads.db"          # SQLite log of every uploaded file

MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "flower_model.h5"
CLASSES_PATH = MODELS_DIR / "class_names.json"
META_PATH = MODELS_DIR / "model_meta.json"
STATE_PATH = MODELS_DIR / "job_state.json"  # shared train/retrain progress

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

# Automatic retraining trigger: retrain when this many new images are pending
RETRAIN_THRESHOLD = int(os.getenv("RETRAIN_THRESHOLD", "50"))
AUTO_RETRAIN = os.getenv("AUTO_RETRAIN", "true").lower() == "true"
# Monitoring trigger: recommend retraining if mean confidence of recent predictions drops below this
LOW_CONFIDENCE = float(os.getenv("LOW_CONFIDENCE", "0.60"))
