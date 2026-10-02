"""Small adapter for custom helmet model prediction calls."""
from config.thresholds import INFERENCE_CONFIDENCE, INFERENCE_IMAGE_SIZE, INFERENCE_IOU


def predict_full_image(model, image):
    """Run the app's standard high-resolution custom-model pass."""
    return model.predict(source=image, conf=INFERENCE_CONFIDENCE,
                         imgsz=INFERENCE_IMAGE_SIZE, iou=INFERENCE_IOU, verbose=False)
