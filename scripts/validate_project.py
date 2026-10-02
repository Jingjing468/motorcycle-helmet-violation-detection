"""Check configured assets, package imports, and optional model initialization."""
import argparse
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import CUSTOM_MODEL_PATH, COCO_MODEL_PATH
from src.models.model_loader import load_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-models", action="store_true", help="Initialize YOLO and EasyOCR")
    args = parser.parse_args()
    required = [CUSTOM_MODEL_PATH, COCO_MODEL_PATH]
    missing = [path for path in required if not path.is_file()]
    if missing:
        parser.error("Missing model files: " + ", ".join(map(str, missing)))
    modules = ["app.ui_components", "src.core.pipeline", "src.models.model_loader",
               "src.visualization.detection_summary", "src.ocr.plate_parser",
               "src.ocr.plate_recognizer"]
    for name in modules:
        importlib.import_module(name)
        print(f"Import OK: {name}")
    if args.load_models:
        model, coco, reader = load_models()
        print(f"Custom model: {'loaded' if model else 'unavailable'}")
        print(f"COCO model: {'loaded' if coco else 'unavailable'}")
        print(f"EasyOCR: {'initialized' if reader else 'unavailable'}")
    else:
        print("Asset and import checks passed. Use --load-models for initialization checks.")


if __name__ == "__main__":
    main()
