"""Normalize manually reviewed project error labels for consistent reporting."""
ERROR_CATEGORIES = {"false helmet detection", "missed no-helmet", "duplicate bike",
                    "missed plate", "unreadable plate"}


def categorize_error(label):
    normalized = " ".join(str(label).strip().lower().replace("_", " ").split())
    if normalized not in ERROR_CATEGORIES:
        raise ValueError(f"Unknown error category: {label}")
    return normalized


def count_errors(labels):
    counts = {category: 0 for category in sorted(ERROR_CATEGORIES)}
    for label in labels:
        counts[categorize_error(label)] += 1
    return counts
