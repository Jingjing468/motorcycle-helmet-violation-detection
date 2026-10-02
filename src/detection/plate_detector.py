"""Plate geometry checks used to retain plausible plate candidates."""
from src.utils.box_utils import box_center


def is_plate_shaped(box, min_aspect=1.1, max_aspect=8.0):
    width, height = max(1, box[2] - box[0]), max(1, box[3] - box[1])
    return min_aspect <= width / height <= max_aspect


def near_motorcycle(box, bikes):
    center_x, center_y = box_center(box)
    for bike in bikes:
        bx1, by1, bx2, by2 = bike["xyxy"]
        if (bx1 - 0.45 * (bx2 - bx1) <= center_x <= bx2 + 0.45 * (bx2 - bx1)
                and by1 - 0.30 * (by2 - by1) <= center_y <= by2 + 0.50 * (by2 - by1)):
            return True
    return False
