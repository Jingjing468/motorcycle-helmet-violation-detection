"""Run the existing Gradio inference callback for one local image."""
import argparse
import sys
from pathlib import Path

import cv2  # type: ignore[import-not-found]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.app import web_detect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--output", default="detection_result.jpg", help="Output image path")
    args = parser.parse_args()
    image = cv2.imread(args.image)
    if image is None:
        parser.error(f"Could not read image: {args.image}")
    result, status, plates, summary, *_gallery = web_detect(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    print(status)
    print(plates)
    print(summary)
    print(f"Saved annotated image: {args.output}")


if __name__ == "__main__":
    main()
