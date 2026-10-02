"""Person candidate filtering used for rider validation."""
from config.thresholds import PERSON_CONFIDENCE


def eligible_person_candidates(detections):
    return [item for item in detections if item["class_name"] == "person" and item["confidence"] >= PERSON_CONFIDENCE]
