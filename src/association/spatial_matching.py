"""Reusable spatial rules for head/person and person/bike matching."""
from src.utils.box_utils import box_center, intersection_area


def head_center_in_person_region(head_box, person_box, padding=(0.20, 0.18, 0.20, 0.36)):
    hx1, hy1, hx2, hy2 = head_box
    px1, py1, px2, py2 = person_box
    pw, ph = max(1, px2 - px1), max(1, py2 - py1)
    region = (px1 - padding[0] * pw, py1 - padding[1] * ph,
              px2 + padding[2] * pw, py1 + padding[3] * ph)
    center = box_center(head_box)
    area = max(1, (hx2 - hx1) * (hy2 - hy1))
    overlap = intersection_area(head_box, region)
    return region[0] <= center[0] <= region[2] and region[1] <= center[1] <= region[3] and overlap / area >= 0.50


def person_bike_distance(person_box, bike_box):
    """Return the existing lower-body proximity score or None when unmatched."""
    px1, py1, px2, py2 = person_box
    bx1, by1, bx2, by2 = bike_box
    bike_w, bike_h = max(1, bx2 - bx1), max(1, by2 - by1)
    person_cx = (px1 + px2) / 2
    horizontal_gap = max(0, bx1 - px2, px1 - bx2)
    x_distance = abs(person_cx - (bx1 + bx2) / 2) / max(0.85 * bike_w, 1)
    lower_body_near = by1 - 0.90 * bike_h <= py2 <= by2 + 1.0 * bike_h
    if not lower_body_near or horizontal_gap > 0.60 * bike_w or x_distance > 1.30:
        return None
    lower_distance = abs(py2 - (by1 + by2) / 2) / bike_h
    return x_distance + lower_distance * 0.25
