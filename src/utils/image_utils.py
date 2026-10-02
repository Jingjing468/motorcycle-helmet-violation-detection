"""Image validation, safe crops, and channel conversion helpers."""
import cv2  # type: ignore[import-not-found]


def is_valid_image(image):
    return image is not None and getattr(image, "ndim", 0) >= 2 and image.size > 0


def safe_crop(image, x1, y1, x2, y2):
    height, width = image.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(width, int(x2)), min(height, int(y2))
    if x2 <= x1 or y2 <= y1:
        return image[0:0, 0:0]
    return image[y1:y2, x1:x2]


def rgb_to_bgr(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def resize(image, width=None, height=None, scale=None):
    if scale is not None:
        return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    if width is None or height is None:
        raise ValueError("Provide width and height or a scale factor")
    return cv2.resize(image, (int(width), int(height)), interpolation=cv2.INTER_CUBIC)
