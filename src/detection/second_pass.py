"""Crop preparation and coordinate mapping for second-pass head inference."""
import cv2  # type: ignore[import-not-found]


def upscale_crop(image, scale=1.5):
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)


def map_scaled_box(box, offset_x, offset_y, scale):
    x1, y1, x2, y2 = box
    return (offset_x + x1 / scale, offset_y + y1 / scale,
            offset_x + x2 / scale, offset_y + y2 / scale)
