"""Metrics helpers for model reports and saved evaluation summaries."""


def precision(true_positive, false_positive):
    denominator = true_positive + false_positive
    return true_positive / denominator if denominator else 0.0


def recall(true_positive, false_negative):
    denominator = true_positive + false_negative
    return true_positive / denominator if denominator else 0.0


def summarize_ultralytics_metrics(metrics):
    """Read standard Ultralytics validation fields without inventing values."""
    box = metrics.box
    return {"precision": float(box.mp), "recall": float(box.mr),
            "map50": float(box.map50), "map50_95": float(box.map)}
