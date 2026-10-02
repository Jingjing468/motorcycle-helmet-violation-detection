"""COCO model adapter for person and motorcycle validation inference."""
from config.thresholds import INFERENCE_IMAGE_SIZE, INFERENCE_IOU, COCO_INFERENCE_CONFIDENCE


def predict_people_and_motorcycles(model, image):
    """Use COCO's person (0) and motorcycle (3) classes at the existing floor."""
    return model.predict(source=image, conf=COCO_INFERENCE_CONFIDENCE,
                         imgsz=INFERENCE_IMAGE_SIZE, iou=INFERENCE_IOU,
                         classes=[0, 3], verbose=False)
