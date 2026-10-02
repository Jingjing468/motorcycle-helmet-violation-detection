"""Rank normalized OCR candidates using confidence and view agreement."""


def score_candidate_entries(entries):
    if not entries:
        return None
    variants = {entry["variant"] for entry in entries}
    mean_confidence = sum(entry["confidence"] for entry in entries) / len(entries)
    mean_position = sum(entry["position"] for entry in entries) / len(entries)
    agreement = min(1.0, len(variants) / 4.0)
    score = 0.50 * mean_confidence + 0.35 * agreement + 0.10 * mean_position
    formats = {}
    for entry in entries:
        formats.setdefault(entry["text"], []).append(entry["confidence"])
    chosen = max(formats, key=lambda text: (len(formats[text]),
                                            sum(formats[text]) / len(formats[text]),
                                            "-" in text))
    return score, mean_confidence, chosen, len(variants)
