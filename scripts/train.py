"""Start a YOLOv8 baseline or fine-tuning run with project settings."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config.paths import COCO_MODEL_PATH


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="YOLO dataset YAML for the intended dataset/split")
    parser.add_argument("--weights", default=str(COCO_MODEL_PATH),
                        help="Starting weights; use the archived custom checkpoint to continue fine-tuning")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--project", default="runs/helmet_violation")
    parser.add_argument("--name", default="experiment")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.is_file():
        parser.error(f"Dataset YAML not found: {data_path}")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    result = model.train(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        epochs=args.epochs,
        project=args.project,
        name=args.name,
        device=args.device,
    )
    save_dir = getattr(result, "save_dir", None)
    print(f"Training finished. Run directory: {save_dir or args.project}")


if __name__ == "__main__":
    main()
