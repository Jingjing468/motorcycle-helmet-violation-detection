"""OCR of detected plate regions, including perspective and text scoring."""
import re

import cv2  # type: ignore[import-not-found]
import numpy as np

from config.thresholds import OCR_CANDIDATE_CONFIDENCE
from src.ocr.easyocr_reader import read_plate
from src.ocr.ocr_scoring import score_candidate_entries
from src.ocr.plate_parser import registration_candidate
from src.ocr.plate_preprocessing import preprocessing_views


def recognize_plates(original, valid_plates, reader):
    """Read each valid plate in left-to-right order; return detections and text."""
    height, width = original.shape[:2]
    plate_detections = sorted(
        valid_plates,
        key=lambda d: (d["xyxy"][0] + d["xyxy"][2]) / 2,
    )
    plate_results = []
    for detection in plate_detections:
        x1, y1, x2, y2 = map(int, detection["xyxy"])
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        print(f"PLATE {len(plate_results) + 1}")
        print(f"Detection confidence: {detection['confidence']:.2f}")
        pad_x = max(4, int(box_w * 0.23))
        pad_top = max(2, int(box_h * 0.18))
        pad_bottom = max(3, int(box_h * 0.23))
        x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_top)
        x2, y2 = min(width, x2 + pad_x), min(height, y2 + pad_bottom)
        x1, x2 = min(x1, width - 1), max(x1 + 1, min(x2, width))
        y1, y2 = min(y1, height - 1), max(y1 + 1, min(y2, height))
        plate_crop = original[y1:y2, x1:x2]
        best_text = detection.get("plate_text", "Unreadable")
        if plate_crop.size:


            def perspective_correct(crop):
                gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(cv2.GaussianBlur(gray_crop, (5, 5), 0), 45, 150)
                contours, _ = cv2.findContours(
                    edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
                )
                crop_area = max(1, crop.shape[0] * crop.shape[1])
                for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:20]:
                    if cv2.contourArea(contour) < 0.20 * crop_area:
                        continue
                    perimeter = cv2.arcLength(contour, True)
                    quad = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
                    if len(quad) != 4 or not cv2.isContourConvex(quad):
                        continue
                    points = quad.reshape(4, 2).astype(np.float32)
                    sums = points.sum(axis=1)
                    differences = np.diff(points, axis=1).reshape(-1)
                    ordered = np.array([
                        points[np.argmin(sums)], points[np.argmin(differences)],
                        points[np.argmax(sums)], points[np.argmax(differences)],
                    ], dtype=np.float32)
                    if len({tuple(point) for point in ordered}) != 4:
                        continue
                    tl, tr, br, bl = ordered
                    out_width = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
                    out_height = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
                    if out_width < 24 or out_height < 8:
                        continue
                    aspect = out_width / out_height
                    if not 1.1 <= aspect <= 8.0:
                        continue
                    if out_width < 0.30 * crop.shape[1] or out_height < 0.25 * crop.shape[0]:
                        continue
                    target = np.array([
                        [0, 0], [out_width - 1, 0],
                        [out_width - 1, out_height - 1], [0, out_height - 1],
                    ], dtype=np.float32)
                    transform = cv2.getPerspectiveTransform(ordered, target)
                    return cv2.warpPerspective(crop, transform, (out_width, out_height)), True
                return crop, False

            perspective_crop, perspective_found = perspective_correct(plate_crop)
            crop_sources = [("Detected crop", plate_crop)]
            if perspective_found:
                crop_sources.append(("Perspective crop", perspective_crop))
                print("Perspective correction: reliable plate quadrilateral found")
            else:
                print("Perspective correction: not applied (no reliable quadrilateral)")

            candidates_by_key = {}
            debug_texts = {}
            allowlist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"

            def add_candidate(text, confidence, variant, position):
                candidate = registration_candidate(text)
                if candidate is None or confidence < OCR_CANDIDATE_CONFIDENCE:
                    return
                normalized = candidate.replace("-", "")
                candidates_by_key.setdefault(normalized, []).append({
                    "text": candidate,
                    "confidence": float(confidence),
                    "variant": variant,
                    "position": float(position),
                })

            for crop_name, source_crop in crop_sources:
                for scale in (4, 6):
                    views = preprocessing_views(source_crop, scale)

                    for view_name, view in views:
                        variant = f"{crop_name} {view_name} {scale}x"
                        ocr_items = read_plate(reader, view, allowlist=allowlist)
                        raw_texts = []
                        line_items = []
                        for polygon, text, confidence in ocr_items:
                            confidence = float(confidence)
                            raw_texts.append(f"{text!r} ({confidence:.2f})")
                            points = [(float(point[0]), float(point[1])) for point in polygon]
                            if not points:
                                continue
                            center_y = sum(point[1] for point in points) / max(1, len(points))
                            box_height = max(
                                1.0,
                                max(point[1] for point in points) - min(point[1] for point in points),
                            )
                            left = min(point[0] for point in points)
                            position = min(1.0, center_y / max(1, view.shape[0]))
                            add_candidate(text, confidence, variant, position)
                            ascii_text = re.sub(r"[^A-Z0-9-]", "", str(text).upper())
                            if ascii_text:
                                line_items.append({
                                    "text": ascii_text,
                                    "confidence": confidence,
                                    "center_y": center_y,
                                    "height": box_height,
                                    "left": left,
                                    "position": position,
                                })
                        debug_texts[variant] = ", ".join(raw_texts) or "no text"

                        # Sort top-to-bottom, group detections on each row,
                        # then combine characters left-to-right within a row.
                        text_lines = []
                        for item in sorted(
                            line_items, key=lambda entry: (entry["center_y"], entry["left"])
                        ):
                            matching_line = next((
                                line for line in text_lines
                                if abs(item["center_y"] - line["center_y"])
                                <= 0.60 * max(item["height"], line["height"])
                            ), None)
                            if matching_line is None:
                                matching_line = {
                                    "center_y": item["center_y"],
                                    "height": item["height"],
                                    "items": [],
                                }
                                text_lines.append(matching_line)
                            matching_line["items"].append(item)
                            matching_line["center_y"] = sum(
                                part["center_y"] for part in matching_line["items"]
                            ) / len(matching_line["items"])
                            matching_line["height"] = max(
                                matching_line["height"], item["height"]
                            )

                        assembled_lines = []
                        for line in sorted(text_lines, key=lambda entry: entry["center_y"]):
                            ordered_items = sorted(line["items"], key=lambda entry: entry["left"])
                            line_text = "".join(item["text"] for item in ordered_items)
                            line_confidence = sum(
                                item["confidence"] for item in ordered_items
                            ) / len(ordered_items)
                            line_position = sum(
                                item["position"] for item in ordered_items
                            ) / len(ordered_items)
                            assembled_lines.append((line_text, line_confidence, line_position))
                            add_candidate(line_text, line_confidence, variant, line_position)
                        if len(assembled_lines) > 1:
                            add_candidate(
                                " ".join(line[0] for line in assembled_lines),
                                sum(line[1] for line in assembled_lines) / len(assembled_lines),
                                variant,
                                sum(line[2] for line in assembled_lines) / len(assembled_lines),
                            )

            ranked_candidates = []
            for normalized, entries in candidates_by_key.items():
                ranked = score_candidate_entries(entries)
                if ranked is not None:
                    score, mean_confidence, chosen_format, support = ranked
                    ranked_candidates.append(
                        (score, mean_confidence, chosen_format, normalized, support)
                    )

            for variant, text in debug_texts.items():
                print(f"{variant} OCR: {text}")
            if ranked_candidates:
                score, confidence, best_text, _, support = max(ranked_candidates)
                print(
                    f"Selected candidate: {best_text} "
                    f"(OCR confidence {confidence:.2f}, agreement {support}/4, score {score:.2f})"
                )
            else:
                print("Selected candidate: Unreadable (no plausible OCR candidate)")
                if "plate_text" not in detection:
                    best_text = "Unreadable"
        else:
            if "plate_text" not in detection:
                best_text = "Unreadable"
            print("Selected candidate: Unreadable (empty plate crop)")
        plate_results.append(best_text)
        print(f"PLATE {len(plate_results)} SELECTED: {best_text}")

    return plate_detections, plate_results
