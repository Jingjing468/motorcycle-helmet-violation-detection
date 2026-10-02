"""Utilities for custom-model helmet/no-helmet candidate selection."""
from config.classes import HEAD_CLASSES


def is_head_detection(class_name):
    return class_name in HEAD_CLASSES


def head_candidates(detections):
    return [item for item in detections if is_head_detection(item["class_name"])]
