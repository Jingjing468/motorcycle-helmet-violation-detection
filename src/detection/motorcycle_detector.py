"""Select motorcycle candidates from custom and COCO detector outputs."""
from config.thresholds import FINAL_CONFIDENCE


def eligible_motorcycle_candidates(detections):
    floor = FINAL_CONFIDENCE["bike"]
    return [item for item in detections if item["class_name"] == "bike" and item["confidence"] >= floor]
