"""Compare known metric dictionaries for two weight checkpoints."""
METRIC_KEYS = ("precision", "recall", "map50", "map50_95")


def compare_metrics(original, fine_tuned):
    """Return fine-tuned minus original metric deltas; inputs must be measured."""
    missing = [key for key in METRIC_KEYS if key not in original or key not in fine_tuned]
    if missing:
        raise ValueError(f"Missing measured metrics: {', '.join(missing)}")
    return {key: float(fine_tuned[key]) - float(original[key]) for key in METRIC_KEYS}
