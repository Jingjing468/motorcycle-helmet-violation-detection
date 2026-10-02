"""Generate the same enhancement views used by the app's EasyOCR stage."""
import cv2  # type: ignore[import-not-found]


def preprocessing_views(crop, scale):
    enlarged = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blurred = cv2.GaussianBlur(clahe, (0, 0), 2.0)
    sharpened = cv2.addWeighted(clahe, 1.7, blurred, -0.7, 0)
    denoised = cv2.fastNlMeansDenoising(clahe, None, h=8, templateWindowSize=7, searchWindowSize=21)
    denoised_sharpened = cv2.addWeighted(denoised, 1.7, cv2.GaussianBlur(denoised, (0, 0), 2.0), -0.7, 0)
    _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    block = min(31, min(clahe.shape))
    if block % 2 == 0:
        block -= 1
    adaptive = cv2.adaptiveThreshold(clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, max(3, block), 11)
    return [("Original", cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB)), ("Grayscale", gray),
            ("CLAHE", clahe), ("Sharpened", sharpened), ("Denoised + sharpened", denoised_sharpened),
            ("Otsu", otsu), ("Inverted Otsu", cv2.bitwise_not(otsu)), ("Adaptive", adaptive)]
