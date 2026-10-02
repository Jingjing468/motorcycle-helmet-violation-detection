"""Geometry-based motorcycle duplicate checks matching the original app."""
from src.utils.box_utils import box_area, box_iou, containment_ratio, box_center


def motorcycle_boxes_duplicate(first, second):
    a, b = first["xyxy"], second["xyxy"]
    overlap_iou = box_iou(a, b)
    containment = containment_ratio(a, b)
    ac, bc = box_center(a), box_center(b)
    center_distance = ((ac[0] - bc[0]) ** 2 + (ac[1] - bc[1]) ** 2) ** 0.5
    close_centers = (overlap_iou > 0 and center_distance <= 0.20 * max(1.0, min(box_area(a) ** 0.5, box_area(b) ** 0.5)))
    return overlap_iou >= 0.45 or close_centers or containment >= 0.60


def merge_motorcycle_detections(items):
    """Merge duplicate candidates, preferring confidence plus source agreement."""
    merged = []
    def quality(item, group_area):
        x1, y1, x2, y2 = item["xyxy"]
        area = max(1, x2 - x1) * max(1, y2 - y1)
        agreement = 0.03 if any(other["source"] != item["source"] and motorcycle_boxes_duplicate(item, other) for other in items) else 0.0
        return item["confidence"] + 0.04 * area / max(1, group_area) + agreement
    for item in sorted(items, key=lambda record: record["confidence"], reverse=True):
        matches = [index for index, kept in enumerate(merged) if motorcycle_boxes_duplicate(item, kept)]
        if not matches:
            merged.append(item)
        elif len(matches) > 1 and any(not motorcycle_boxes_duplicate(merged[a], merged[b])
                                      for offset, a in enumerate(matches) for b in matches[offset + 1:]):
            merged.append(item)
        else:
            for index in matches:
                kept = merged[index]
                if quality(item, max(box_area(item["xyxy"]), box_area(kept["xyxy"]))) > quality(kept, max(box_area(item["xyxy"]), box_area(kept["xyxy"]))):
                    merged[index] = item
    return merged
