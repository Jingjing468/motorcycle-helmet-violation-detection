"""Run Ultralytics validation when a dataset YAML is supplied."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import CUSTOM_MODEL_PATH
from src.evaluation.metrics import summarize_ultralytics_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Dataset YAML path")
    parser.add_argument("--weights", default=str(CUSTOM_MODEL_PATH))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    from ultralytics import YOLO
    metrics = YOLO(args.weights).val(data=args.data, imgsz=args.imgsz, device=args.device)
    for name, value in summarize_ultralytics_metrics(metrics).items():
        print(f"{name}: {value:.4f}")


if __name__ == "__main__":
    main()
