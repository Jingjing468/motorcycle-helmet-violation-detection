"""Load each neural model once for the lifetime of the Gradio process."""
import importlib

from config.paths import COCO_MODEL_PATH, CUSTOM_MODEL_PATH
from config.settings import OCR_LANGUAGES

_loaded_models = None


def load_models():
    """Return cached (custom YOLO, COCO YOLO, EasyOCR reader) instances."""
    global _loaded_models
    if _loaded_models is not None:
        return _loaded_models
    try:
        from ultralytics import YOLO
    except ImportError:  # preserve a usable UI when optional runtime is absent
        YOLO = None
    try:
        easyocr = importlib.import_module("easyocr")
    except ImportError:
        easyocr = None
    model = YOLO(str(CUSTOM_MODEL_PATH)) if YOLO is not None else None
    if model is not None:
        print("MODEL CLASSES:")
        print(model.names)
    try:
        coco_model = YOLO(str(COCO_MODEL_PATH)) if YOLO is not None else None
    except Exception as exc:
        print(f"COCO validation model unavailable: {exc}")
        coco_model = None
    reader = easyocr.Reader(OCR_LANGUAGES) if easyocr is not None else None
    _loaded_models = model, coco_model, reader
    return _loaded_models
