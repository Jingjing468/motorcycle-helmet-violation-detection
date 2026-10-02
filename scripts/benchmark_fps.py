"""Measure end-to-end image inference time and FPS on supplied images."""
import argparse
import sys
import time
from pathlib import Path

import cv2  # type: ignore[import-not-found]

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import TEST_IMAGES_DIR
from src.utils.file_utils import image_files
from app.app import web_detect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", default=str(TEST_IMAGES_DIR), help="Image file or folder")
    args = parser.parse_args()
    path = Path(args.images)
    images = [path] if path.is_file() else image_files(path)
    if not images:
        parser.error("No benchmark images found; supply --images with at least one image")
    elapsed = []
    for image_path in images:
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            print(f"Skipping unreadable image: {image_path}")
            continue
        start = time.perf_counter()
        web_detect(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        elapsed.append(time.perf_counter() - start)
    if not elapsed:
        parser.error("No readable images were available")
    total = sum(elapsed)
    print(f"Images: {len(elapsed)} | Total: {total:.3f}s | Mean: {total / len(elapsed):.3f}s/image | FPS: {len(elapsed) / total:.3f}")


if __name__ == "__main__":
    main()
