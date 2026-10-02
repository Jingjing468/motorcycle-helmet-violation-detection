"""Resolve per-person helmet class candidates using the original decision rule."""
from config.thresholds import HELMET_DECISION_CONFIDENCE, NO_HELMET_DECISION_CONFIDENCE


def select_head_class(person_heads):
    helmet = max((h for h in person_heads if h["class_name"] == "helmet"),
                 key=lambda h: h["confidence"], default=None)
    no_helmet = max((h for h in person_heads if h["class_name"] == "no-helmet"),
                    key=lambda h: h["confidence"], default=None)
    helmet_conf = helmet["confidence"] if helmet else 0.0
    no_helmet_conf = no_helmet["confidence"] if no_helmet else 0.0
    if helmet and helmet_conf >= HELMET_DECISION_CONFIDENCE and helmet_conf > no_helmet_conf:
        return helmet
    if no_helmet and no_helmet_conf >= NO_HELMET_DECISION_CONFIDENCE:
        return no_helmet
    return None
