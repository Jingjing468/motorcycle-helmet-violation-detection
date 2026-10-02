"""Shared confidence filters sourced from project configuration."""
from config.thresholds import FINAL_CONFIDENCE


def above_threshold(detection, threshold):
    return detection["confidence"] >= threshold


def accepted_final_detection(detection):
    return above_threshold(detection, FINAL_CONFIDENCE.get(detection["class_name"], 1.0))
