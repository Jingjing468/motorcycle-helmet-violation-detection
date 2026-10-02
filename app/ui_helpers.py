"""Formatting helpers for output fields outside the detection summary card."""


def format_violation_status(violation):
    return "🚨 HELMET VIOLATION DETECTED" if violation else "✅ NO VIOLATION DETECTED"


def format_plate_results(plate_detections, plate_results):
    if not plate_detections:
        return "License Plates Detected: 0\nNot detected"
    return "License Plates Detected: {}\n{}".format(
        len(plate_results),
        "\n".join(
            f"{index}. {text} — detection {detection['confidence'] * 100:.0f}%"
            for index, (text, detection) in enumerate(zip(plate_results, plate_detections), 1)
        ),
    )
