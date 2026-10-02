"""Class-aware OpenCV non-maximum suppression."""
import cv2  # type: ignore[import-not-found]


def class_aware_nms(items, iou_threshold=0.50):
    """Keep per-class boxes selected by OpenCV NMS, preserving item records."""
    merged = []
    for class_name in dict.fromkeys(item["class_name"] for item in items):
        detections = [item for item in items if item["class_name"] == class_name]
        boxes, scores = [], []
        for item in detections:
            x1, y1, x2, y2 = item["xyxy"]
            boxes.append([int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1))])
            scores.append(item["confidence"])
        kept = cv2.dnn.NMSBoxes(boxes, scores, 0.0, iou_threshold)
        for index in kept.flatten().tolist() if len(kept) else []:
            merged.append(detections[index])
    return merged
