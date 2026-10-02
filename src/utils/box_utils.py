"""Pure geometry helpers for xyxy boxes."""


def box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def intersection_area(first, second):
    return max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1]))


def box_iou(first, second):
    overlap = intersection_area(first, second)
    union = box_area(first) + box_area(second) - overlap
    return overlap / union if union else 0.0


def containment_ratio(first, second):
    smaller_area = min(box_area(first), box_area(second))
    return intersection_area(first, second) / smaller_area if smaller_area else 0.0


def clip_box(box, width, height):
    x1, y1, x2, y2 = box
    return max(0.0, min(width, x1)), max(0.0, min(height, y1)), max(0.0, min(width, x2)), max(0.0, min(height, y2))


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return x1, y1, x2 - x1, y2 - y1
