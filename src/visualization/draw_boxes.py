"""Draw validated public detections without drawing internal person boxes."""
import cv2  # type: ignore[import-not-found]

from src.utils.image_utils import bgr_to_rgb
from src.visualization.colors import COLORS_BGR, DISPLAY_NAMES


def draw_detections(image, detections):
    """Draw boxes and confidence labels using the original placement rules."""
    detected_image = image.copy()
    height, width = detected_image.shape[:2]
    placed_labels = []
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection["xyxy"])
        cv2.rectangle(detected_image, (x1, y1), (x2, y2),
                      COLORS_BGR.get(detection["class_name"], (0, 220, 255)), 2)
    for detection in sorted(detections, key=lambda item: (item["xyxy"][1], item["xyxy"][0])):
        x1, y1, x2, y2 = map(int, detection["xyxy"])
        class_name = detection["class_name"]
        color = COLORS_BGR.get(class_name, (0, 220, 255))
        label = f"{DISPLAY_NAMES.get(class_name, class_name)} {detection['confidence'] * 100:.0f}%"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        label_width = min(width, text_width + 8)
        label_height = text_height + baseline + 6
        label_x = min(max(0, x1), max(0, width - label_width))
        candidate_ys = [y1 - label_height, y2 + 2, y1 + 2]
        candidate_ys.extend(y1 - label_height - step * (label_height + 2) for step in range(1, 5))
        label_y = min(max(0, y1 - label_height), max(0, height - label_height))
        for candidate_y in candidate_ys:
            candidate_y = min(max(0, candidate_y), max(0, height - label_height))
            candidate_rect = (label_x, candidate_y, label_x + label_width, candidate_y + label_height)
            overlaps = any(candidate_rect[0] < old[2] and candidate_rect[2] > old[0]
                           and candidate_rect[1] < old[3] and candidate_rect[3] > old[1]
                           for old in placed_labels)
            if not overlaps:
                label_y = candidate_y
                break
        cv2.rectangle(detected_image, (label_x, label_y),
                      (label_x + label_width, label_y + label_height), color, -1)
        cv2.putText(detected_image, label, (label_x + 4, label_y + text_height + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        placed_labels.append((label_x, label_y, label_x + label_width, label_y + label_height))
    return bgr_to_rgb(detected_image)
