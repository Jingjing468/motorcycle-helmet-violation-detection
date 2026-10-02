"""Filesystem locations used by the application and command-line tools."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "model"
RESULTS_DIR = PROJECT_ROOT / "results"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"
CUSTOM_MODEL_PATH = MODEL_DIR / "helmet_best.pt"
COCO_MODEL_PATH = MODEL_DIR / "coco" / "yolov8n.pt"
ARCHIVED_MODEL_PATH = MODEL_DIR / "archive" / "helmet_best_old.pt"
