import importlib
import re
from pathlib import Path

import gradio as gr  # type: ignore[import-not-found]
import cv2  # type: ignore[import-not-found]
import numpy as np

try:
    easyocr = importlib.import_module("easyocr")
except ImportError:  # pragma: no cover - optional dependency for OCR path
    easyocr = None

try:
    from ultralytics import YOLO  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - optional dependency for model path
    YOLO = None

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "model" / "helmet_best.pt"
COCO_MODEL_PATH = PROJECT_ROOT / "model" / "coco" / "yolov8n.pt"

# Load Deep Learning model
model = YOLO(str(MODEL_PATH)) if YOLO is not None else None
if model is not None:
    print("MODEL CLASSES:")
    print(model.names)

# COCO model supplies internal person validation and motorcycle detections.
# Use the checked-in COCO weights for person and motorcycle validation.
try:
    coco_model = YOLO(str(COCO_MODEL_PATH)) if YOLO is not None else None
except Exception as exc:  # keep the app available if validator weights cannot load
    print(f"COCO validation model unavailable: {exc}")
    coco_model = None

# Load OCR
reader = easyocr.Reader(["en"]) if easyocr is not None else None


def web_detect(image):
    if image is None:
        return None, "Please upload an image", "Not detected", ""

    if model is None:
        return None, "Model dependency is missing. Please install ultralytics.", "Not detected", ""

    if reader is None:
        return None, "OCR dependency is missing. Please install easyocr.", "Not detected", ""

    original = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = original.shape[:2]

    # Run the normal high-resolution detector on the full image first.
    # A low initial confidence retains candidates; class thresholds below
    # decide which detections are accepted.
    prediction_options = {"conf": 0.08, "imgsz": 1280, "iou": 0.50, "verbose": False}
    full_results = model.predict(source=original, **prediction_options)
    print("CUSTOM RAW:")
    for result in full_results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            print(f"{str(model.names[class_id]).lower()} {float(box.conf[0]):.2f}")

    # Tile only large images to keep normal image inference responsive.
    # Crop-local bike/plate boxes are translated back to full-image coordinates.
    crop_results = []
    if width * height >= 1_500_000 or max(width, height) >= 1800:
        crop_w = max(1, int(width * 0.60))
        crop_h = max(1, int(height * 0.60))
        x_starts = sorted(set((0, max(0, width - crop_w))))
        y_starts = sorted(set((0, max(0, height - crop_h))))
        for y0 in y_starts:
            for x0 in x_starts:
                crop = original[y0:min(height, y0 + crop_h), x0:min(width, x0 + crop_w)]
                if crop.size == 0 or (crop.shape[0] == height and crop.shape[1] == width):
                    continue
                for result in model.predict(source=crop, **prediction_options):
                    if result.boxes is None:
                        continue
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        class_name = str(model.names[class_id]).lower()
                        if class_name not in ("bike", "number-plate"):
                            continue
                        x1, y1, x2, y2 = map(float, box.xyxy[0])
                        crop_results.append({
                            "class_id": class_id,
                            "class_name": class_name,
                            "confidence": float(box.conf[0]),
                            "xyxy": (x1 + x0, y1 + y0, x2 + x0, y2 + y0),
                            "source": "tiled-crop",
                        })

    # Raw candidates use the inference floor; final displayed objects use
    # stronger class-specific thresholds after person/bike validation.
    final_confidence_thresholds = {
        "bike": 0.20,
        "helmet": 0.30,
        "no-helmet": 0.30,
        "number-plate": 0.08,
    }

    # Apply class-specific confidence thresholds after both prediction passes.
    # Crop detections and full-image detections are then merged by class with
    # IoU NMS so the same bike or plate is not reported several times.
    candidates = []
    for result in full_results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = str(model.names[class_id]).lower()
            confidence = float(box.conf[0])
            if confidence >= 0.08:
                candidates.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "xyxy": tuple(map(float, box.xyxy[0])),
                    "source": "full-image",
                })
    candidates.extend(
        detection for detection in crop_results
        if detection["confidence"] >= 0.08
    )
    full_image_head_candidates = [
        detection for detection in candidates
        if detection["source"] == "full-image"
        and detection["class_name"] in ("helmet", "no-helmet")
    ]

    def merge_duplicate_detections(items):
        """Apply class-aware NMS to merge boxes from full and crop passes."""
        merged = []
        for class_name in dict.fromkeys(d["class_name"] for d in items):
            class_detections = [d for d in items if d["class_name"] == class_name]
            boxes_xywh, scores = [], []
            for detection in class_detections:
                x1, y1, x2, y2 = detection["xyxy"]
                boxes_xywh.append([
                    int(x1), int(y1), max(1, int(x2 - x1)), max(1, int(y2 - y1))
                ])
                scores.append(detection["confidence"])
            kept = cv2.dnn.NMSBoxes(boxes_xywh, scores, 0.0, 0.50)
            for index in kept.flatten().tolist() if len(kept) else []:
                merged.append(class_detections[index])
        return merged

    def motorcycle_boxes_duplicate(first, second):
        """Match duplicate motorcycle boxes while protecting nearby bikes."""
        ax1, ay1, ax2, ay2 = first["xyxy"]
        bx1, by1, bx2, by2 = second["xyxy"]
        intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
            0, min(ay2, by2) - max(ay1, by1)
        )
        area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
        area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
        iou = intersection / max(1, area_a + area_b - intersection)
        containment = intersection / max(1, min(area_a, area_b))
        center_distance = (
            ((ax1 + ax2 - bx1 - bx2) / 2) ** 2
            + ((ay1 + ay2 - by1 - by2) / 2) ** 2
        ) ** 0.5
        smaller_box_scale = min(area_a ** 0.5, area_b ** 0.5)
        close_centers = (
            intersection > 0 and center_distance <= 0.20 * max(1.0, smaller_box_scale)
        )
        nested_box = containment >= 0.60
        return iou >= 0.45 or close_centers or nested_box

    def merge_motorcycle_detections(items):
        """Keep one existing source box per motorcycle without averaging."""
        merged = []
        duplicate_group = 0

        def box_area(detection):
            x1, y1, x2, y2 = detection["xyxy"]
            return max(1, x2 - x1) * max(1, y2 - y1)

        def source_agreement(detection):
            return any(
                other["source"] != detection["source"]
                and motorcycle_boxes_duplicate(detection, other)
                for other in items
            )

        def quality(detection, group_area):
            # Confidence is primary. Slight tie-breakers favor complete boxes
            # and agreement between custom and COCO detections.
            coverage = box_area(detection) / max(1, group_area)
            agreement = 0.03 if source_agreement(detection) else 0.0
            return detection["confidence"] + 0.04 * coverage + agreement

        ordered = sorted(items, key=lambda item: item["confidence"], reverse=True)
        print("BIKE RAW:")
        for detection in ordered:
            print(
                f"{detection['source']}: {detection['confidence']:.2f} "
                f"{tuple(round(value) for value in detection['xyxy'])}"
            )

        for detection in ordered:
            bridge_peers = [
                other for other in items
                if other is not detection and motorcycle_boxes_duplicate(detection, other)
            ]
            if len(bridge_peers) > 1 and any(
                not motorcycle_boxes_duplicate(bridge_peers[first], bridge_peers[second])
                for first in range(len(bridge_peers))
                for second in range(first + 1, len(bridge_peers))
            ):
                print(
                    f"REJECTING broad Bike box {detection['confidence']:.2f} "
                    "because it overlaps distinct motorcycles"
                )
                continue
            matched_indexes = [
                index for index, kept in enumerate(merged)
                if motorcycle_boxes_duplicate(detection, kept)
            ]
            if not matched_indexes:
                merged.append(detection)
                continue

            # A broad box may cover two genuinely separate neighboring bikes.
            if len(matched_indexes) > 1 and any(
                not motorcycle_boxes_duplicate(merged[first], merged[second])
                for offset, first in enumerate(matched_indexes)
                for second in matched_indexes[offset + 1:]
            ):
                merged.append(detection)
                continue

            duplicate_group += 1
            for index in matched_indexes:
                previous = merged[index]
                group_area = max(box_area(previous), box_area(detection))
                if quality(detection, group_area) > quality(previous, group_area):
                    merged[index] = detection
                    previous = detection
                print(
                    f"DUPLICATE GROUP {duplicate_group} -> keeping Bike "
                    f"{previous['confidence']:.2f} ({previous['source']})"
                )
        return merged

    raw_bike_candidates = [d for d in candidates if d["class_name"] == "bike"]
    raw_detections = merge_duplicate_detections(
        [d for d in candidates if d["class_name"] != "bike"]
    )

    # The COCO model detects people and motorcycles. Tile only large images so
    # small distant motorcycles benefit without slowing ordinary-sized photos.
    persons = []
    coco_motorcycle_candidates = []
    if coco_model is not None:
        coco_options = {"conf": 0.20, "imgsz": 1280, "iou": 0.50, "classes": [0, 3], "verbose": False}
        coco_results = coco_model.predict(source=original, **coco_options)
        coco_tiles = []
        if width * height >= 1_500_000 or max(width, height) >= 1800:
            tile_w, tile_h = max(1, int(width * 0.60)), max(1, int(height * 0.60))
            tile_x = sorted(set((0, max(0, width - tile_w))))
            tile_y = sorted(set((0, max(0, height - tile_h))))
            for y0 in tile_y:
                for x0 in tile_x:
                    tile = original[y0:min(height, y0 + tile_h), x0:min(width, x0 + tile_w)]
                    if tile.size and tile.shape[:2] != original.shape[:2]:
                        for result in coco_model.predict(source=tile, **coco_options):
                            coco_tiles.append((result, x0, y0))

        print("COCO RAW:")
        coco_sources = [(result, 0, 0) for result in coco_results] + coco_tiles
        raw_person_candidates = []
        for result, offset_x, offset_y in coco_sources:
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = {0: "person", 3: "motorcycle"}.get(class_id)
                confidence = float(box.conf[0])
                if class_name is None:
                    continue
                print(f"{class_name.title()} {confidence:.2f}")
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                candidate = {
                    "class_id": class_id,
                    "class_name": "bike" if class_name == "motorcycle" else "person",
                    "confidence": confidence,
                    "xyxy": (x1 + offset_x, y1 + offset_y, x2 + offset_x, y2 + offset_y),
                    "source": "coco-motorcycle" if class_name == "motorcycle" else "coco-person",
                }
                if class_name == "person" and confidence >= 0.25:
                    raw_person_candidates.append(candidate)
                elif class_name == "motorcycle" and confidence >= 0.20:
                    coco_motorcycle_candidates.append(candidate)

        persons = merge_duplicate_detections(raw_person_candidates)

    # Merge custom bike boxes and COCO motorcycle boxes class-wise. COCO can
    # contribute a bike even when the custom model missed it.
    raw_bike_candidates.extend(coco_motorcycle_candidates)
    custom_bikes = [
        d for d in raw_bike_candidates
        if d["class_name"] == "bike"
        and d["confidence"] >= final_confidence_thresholds["bike"]
    ]
    bikes = merge_motorcycle_detections(custom_bikes + coco_motorcycle_candidates)
    print(f"FINAL BIKES: {len(bikes)}")
    raw_detections.extend(bikes)

    # Dedicated full-image plate inference runs below the general detection
    # floor, before app-level confidence or geometry filtering.
    plate_prediction_options = {"imgsz": 1280, "conf": 0.03, "iou": 0.50, "verbose": False}
    full_plate_candidates = []
    full_plate_results = model.predict(source=original, **plate_prediction_options)
    print("RAW PLATES:")
    for result in full_plate_results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = str(model.names[class_id])
            if class_name != "number-plate":
                continue
            confidence = float(box.conf[0])
            xyxy = tuple(map(float, box.xyxy[0]))
            print(f"number-plate confidence = {confidence:.2f}, bbox = {xyxy}")
            full_plate_candidates.append({
                "class_id": class_id,
                "class_name": "number-plate",
                "confidence": confidence,
                "xyxy": xyxy,
                "source": "full-image-plate-pass",
            })

    # Search the lower/rear area of every merged motorcycle for plates that
    # the full-image pass may miss; crop-local coordinates are mapped back.
    motorcycle_plate_candidates = []
    for bike_index, bike in enumerate(bikes, start=1):
        bx1, by1, bx2, by2 = bike["xyxy"]
        bike_w, bike_h = max(1, bx2 - bx1), max(1, by2 - by1)
        crop_x1 = max(0, int(bx1 - 0.20 * bike_w))
        crop_x2 = min(width, int(bx2 + 0.20 * bike_w))
        crop_y1 = max(0, int(by1 + 0.30 * bike_h))
        crop_y2 = min(height, int(by2 + 0.35 * bike_h))
        plate_crop = original[crop_y1:crop_y2, crop_x1:crop_x2]
        if plate_crop.size == 0:
            continue
        print(f"SECOND PASS PLATES (motorcycle {bike_index}):")
        for result in model.predict(source=plate_crop, **plate_prediction_options):
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = str(model.names[class_id])
                if class_name != "number-plate":
                    continue
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                mapped_box = (
                    x1 + crop_x1, y1 + crop_y1,
                    x2 + crop_x1, y2 + crop_y1,
                )
                print(f"number-plate {confidence:.2f} bbox={mapped_box}")
                motorcycle_plate_candidates.append({
                    "class_id": class_id,
                    "class_name": "number-plate",
                    "confidence": confidence,
                    "xyxy": mapped_box,
                    "source": "motorcycle-plate-pass",
                })

    # Combine every raw plate source, then apply the requested 0.08 threshold.
    # A 0.03-0.08 candidate can remain only when it has a plate-like shape and
    # sits inside or close to a detected motorcycle region.
    existing_plate_candidates = [
        d for d in raw_detections if d["class_name"] == "number-plate"
    ]
    raw_plate_candidates = (
        existing_plate_candidates + full_plate_candidates + motorcycle_plate_candidates
    )
    valid_plate_candidates = []
    for plate in raw_plate_candidates:
        x1, y1, x2, y2 = plate["xyxy"]
        box_w, box_h = max(1, x2 - x1), max(1, y2 - y1)
        aspect_ratio = box_w / box_h
        plate_shaped = 1.1 <= aspect_ratio <= 8.0
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        near_bike = any(
            bx1 - 0.45 * (bx2 - bx1) <= center_x <= bx2 + 0.45 * (bx2 - bx1)
            and by1 - 0.30 * (by2 - by1) <= center_y <= by2 + 0.50 * (by2 - by1)
            for bx1, by1, bx2, by2 in (bike["xyxy"] for bike in bikes)
        )
        if plate["confidence"] >= final_confidence_thresholds["number-plate"]:
            valid_plate_candidates.append(plate)
        elif plate["confidence"] >= 0.03 and near_bike and plate_shaped:
            valid_plate_candidates.append(plate)

    final_plate_detections = merge_duplicate_detections(valid_plate_candidates)
    print("FINAL PLATES:")
    for plate_index, plate in enumerate(final_plate_detections, start=1):
        print(f"Plate {plate_index} {plate['confidence']:.2f}")
    raw_detections = [
        d for d in raw_detections if d["class_name"] != "number-plate"
    ] + final_plate_detections

    def person_bike_distance(person, bike):
        """Measure generous person-to-motorcycle proximity using the lower body."""
        px1, py1, px2, py2 = person["xyxy"]
        bx1, by1, bx2, by2 = bike["xyxy"]
        bike_w, bike_h = max(1, bx2 - bx1), max(1, by2 - by1)
        person_cx = (px1 + px2) / 2
        horizontal_gap = max(0, bx1 - px2, px1 - bx2)
        x_distance = abs(person_cx - (bx1 + bx2) / 2) / max(0.85 * bike_w, 1)
        lower_body_near = by1 - 0.90 * bike_h <= py2 <= by2 + 1.0 * bike_h
        overlaps_horizontally = horizontal_gap <= 0.60 * bike_w
        lower_distance = abs(py2 - (by1 + by2) / 2) / bike_h
        if not lower_body_near or not overlaps_horizontally or x_distance > 1.30:
            return None
        return x_distance + lower_distance * 0.25

    # Inspect an enlarged crop around each detected motorcycle and its upper
    # occupant area. This bike-specific pass focuses on small helmet labels.
    occupant_head_candidates = []
    second_pass_thresholds = {"helmet": 0.15, "no-helmet": 0.15}
    for bike_index, bike in enumerate(bikes, start=1):
        bx1, by1, bx2, by2 = bike["xyxy"]
        bike_w, bike_h = max(1, bx2 - bx1), max(1, by2 - by1)
        crop_x1 = max(0, int((bx1 + bx2) / 2 - 0.90 * bike_w))
        crop_x2 = min(width, int((bx1 + bx2) / 2 + 0.90 * bike_w))
        crop_y1 = max(0, int(by1 - 1.60 * bike_h))
        crop_y2 = min(height, int(by2 + 0.15 * bike_h))
        occupant_crop = original[crop_y1:crop_y2, crop_x1:crop_x2]
        if occupant_crop.size == 0:
            continue

        # Upscaling gives small driver and passenger heads more pixels before
        # YOLO inference; prediction boxes are scaled back to image coordinates.
        # Extend the crop to include any nearby COCO person box, allowing
        # passengers slightly beside or behind the motorcycle to be scanned.
        for person in persons:
            if person_bike_distance(person, bike) is not None:
                px1, py1, px2, py2 = person["xyxy"]
                crop_x1 = max(0, min(crop_x1, int(px1)))
                crop_x2 = min(width, max(crop_x2, int(px2)))
                crop_y1 = max(0, min(crop_y1, int(py1)))
                crop_y2 = min(height, max(crop_y2, int(py2)))
        occupant_crop = original[crop_y1:crop_y2, crop_x1:crop_x2]
        if occupant_crop.size == 0:
            continue
        scale = 1.5
        enlarged_crop = cv2.resize(
            occupant_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
        )
        print(f"RAW SECOND PASS (bike {bike_index}):")
        for result in model.predict(
            source=enlarged_crop, conf=0.08, imgsz=1280, iou=0.50, verbose=False
        ):
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = str(model.names[class_id]).lower()
                if class_name not in ("helmet", "no-helmet"):
                    continue
                confidence = float(box.conf[0])
                print(f"{class_name} {confidence:.2f}")
                if confidence < second_pass_thresholds[class_name]:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                occupant_head_candidates.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "xyxy": (
                        crop_x1 + x1 / scale, crop_y1 + y1 / scale,
                        crop_x1 + x2 / scale, crop_y1 + y2 / scale,
                    ),
                    "source": "second-pass",
                })

    # For each person associated with a motorcycle, run a focused head crop
    # only when full-image inference found neither head class for that person.
    associated_people = []
    for person_index, person in enumerate(persons):
        bike_matches = [
            (distance, bike_index)
            for bike_index, bike in enumerate(bikes)
            if (distance := person_bike_distance(person, bike)) is not None
        ]
        if bike_matches:
            _, bike_index = min(bike_matches)
            associated_people.append((person_index, person, bike_index))

    def head_center_in_person_region(head, person):
        hx1, hy1, hx2, hy2 = head["xyxy"]
        px1, py1, px2, py2 = person["xyxy"]
        head_cx, head_cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
        person_h = max(1, py2 - py1)
        return px1 <= head_cx <= px2 and py1 <= head_cy <= py1 + 0.45 * person_h

    person_head_thresholds = {"helmet": 0.20, "no-helmet": 0.15}
    for person_index, person, bike_index in associated_people:
        full_person_heads = [
            head for head in full_image_head_candidates
            if head_center_in_person_region(head, person)
        ]
        print(f"PERSON {person_index + 1} -> BIKE {bike_index + 1}")
        print("FULL IMAGE:")
        for class_name in ("helmet", "no-helmet"):
            class_heads = [h for h in full_person_heads if h["class_name"] == class_name]
            value = ", ".join(f"{h['confidence']:.2f}" for h in class_heads) or "none"
            print(f"{class_name}: {value}")
        if full_person_heads:
            continue

        px1, py1, px2, py2 = person["xyxy"]
        person_w, person_h = max(1, px2 - px1), max(1, py2 - py1)
        head_x1 = max(0, int(px1 - 0.15 * person_w))
        head_x2 = min(width, int(px2 + 0.15 * person_w))
        head_y1 = max(0, int(py1 - 0.12 * person_h))
        head_y2 = min(height, int(py1 + 0.45 * person_h))
        head_crop = original[head_y1:head_y2, head_x1:head_x2]
        if head_crop.size == 0:
            continue

        head_scale = 2.0
        enlarged_head_crop = cv2.resize(
            head_crop, None, fx=head_scale, fy=head_scale,
            interpolation=cv2.INTER_CUBIC
        )
        print("HEAD SECOND PASS:")
        for result in model.predict(
            source=enlarged_head_crop, imgsz=960, conf=0.05,
            iou=0.50, verbose=False
        ):
            if result.boxes is None:
                continue
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = str(model.names[class_id]).lower()
                if class_name not in ("helmet", "no-helmet"):
                    continue
                confidence = float(box.conf[0])
                print(f"{class_name} {confidence:.2f}")
                if confidence < person_head_thresholds[class_name]:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                occupant_head_candidates.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "xyxy": (
                        head_x1 + x1 / head_scale, head_y1 + y1 / head_scale,
                        head_x1 + x2 / head_scale, head_y1 + y2 / head_scale,
                    ),
                    "source": "person-head-fallback",
                    "person_index": person_index,
                })

    # Merge bike-level and person-head second-pass detections with full-image
    # predictions before final validation and duplicate suppression.
    raw_detections = merge_duplicate_detections(raw_detections + occupant_head_candidates)

    raw_head_detections = [
        d for d in raw_detections
        if d["class_name"] in ("helmet", "no-helmet")
    ]
    print("RAW:")
    for head in raw_head_detections:
        print(f"{head['class_name'].replace('-', ' ').title()} {head['confidence']:.2f}")

    def find_person_for_head(head):
        """Require the head center to be within a person's upper 45 percent."""
        hx1, hy1, hx2, hy2 = head["xyxy"]
        head_cx, head_cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
        matches = []
        for person_index, person in enumerate(persons):
            px1, py1, px2, py2 = person["xyxy"]
            person_h = max(1, py2 - py1)
            if px1 <= head_cx <= px2 and py1 <= head_cy <= py1 + 0.45 * person_h:
                # Prefer confident person boxes and heads near the upper center.
                vertical_fraction = (head_cy - py1) / person_h
                matches.append((-(person["confidence"] - vertical_fraction * 0.1), person_index))
        return min(matches)[1] if matches else None

    def has_second_pass_support(head):
        """Check for same-class second-pass evidence at the same head location."""
        hx1, hy1, hx2, hy2 = head["xyxy"]
        hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
        hw, hh = max(1, hx2 - hx1), max(1, hy2 - hy1)
        for candidate in occupant_head_candidates:
            if candidate["class_name"] != head["class_name"]:
                continue
            x1, y1, x2, y2 = candidate["xyxy"]
            ccx, ccy = (x1 + x2) / 2, (y1 + y2) / 2
            center_distance = (
                ((hcx - ccx) / max(hw, x2 - x1, 1)) ** 2
                + ((hcy - ccy) / max(hh, y2 - y1, 1)) ** 2
            ) ** 0.5
            if center_distance <= 0.35:
                return True
        return False

    def has_person_head_fallback_support(head):
        """Identify a matching detection from the person-specific head crop."""
        hx1, hy1, hx2, hy2 = head["xyxy"]
        hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
        hw, hh = max(1, hx2 - hx1), max(1, hy2 - hy1)
        for candidate in occupant_head_candidates:
            if (candidate.get("source") != "person-head-fallback"
                    or candidate["class_name"] != head["class_name"]):
                continue
            x1, y1, x2, y2 = candidate["xyxy"]
            ccx, ccy = (x1 + x2) / 2, (y1 + y2) / 2
            distance = (
                ((hcx - ccx) / max(hw, x2 - x1, 1)) ** 2
                + ((hcy - ccy) / max(hh, y2 - y1, 1)) ** 2
            ) ** 0.5
            if distance <= 0.35:
                return True
        return False

    # Validate head -> person -> motorcycle. Weak custom-model scores survive
    # only with strong person/bike geometry and matching second-pass support.
    validated_head_candidates = []
    for head in raw_head_detections:
        person_index = find_person_for_head(head)
        label = head["class_name"].replace("-", " ").title()
        if person_index is None:
            print(f"REJECTED: {label} {head['confidence']:.2f} -> no matching person")
            continue
        person = persons[person_index]
        bike_matches = [
            (distance, bike_index)
            for bike_index, bike in enumerate(bikes)
            if (distance := person_bike_distance(person, bike)) is not None
        ]
        if not bike_matches:
            print(f"REJECTED: {label} {head['confidence']:.2f} -> person not associated with a bike")
            continue
        bike_distance, bike_index = min(bike_matches)
        second_pass_support = has_second_pass_support(head)
        person_head_support = has_person_head_fallback_support(head)
        normal_confidence = final_confidence_thresholds[head["class_name"]]
        strong_existing_support = (
            person["confidence"] >= 0.50
            and bike_distance <= 0.75
            and second_pass_support
        )
        accepted_head_fallback = (
            person_head_support
            and head["confidence"] >= person_head_thresholds[head["class_name"]]
        )
        accepted_existing_fallback = (
            strong_existing_support
            and head["confidence"] >= second_pass_thresholds[head["class_name"]]
        )
        if (head["confidence"] < normal_confidence
                and not (accepted_head_fallback or accepted_existing_fallback)):
            print(f"REJECTED: {label} {head['confidence']:.2f} -> below final threshold without strong second-pass support")
            continue
        validated_head_candidates.append({
            **head,
            "person_index": person_index,
            "person_confidence": person["confidence"],
            "bike_index": bike_index,
            "bike_distance": bike_distance,
            "second_pass_support": second_pass_support or person_head_support,
        })

    # Resolve opposite labels only for the same person and same head geometry.
    validated_heads = []
    uncertain_heads = []
    for person_index in dict.fromkeys(h["person_index"] for h in validated_head_candidates):
        person_heads = [h for h in validated_head_candidates if h["person_index"] == person_index]
        active = set(range(len(person_heads)))
        for i, first in enumerate(person_heads):
            if i not in active:
                continue
            ax1, ay1, ax2, ay2 = first["xyxy"]
            acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
            aw, ah = max(1, ax2 - ax1), max(1, ay2 - ay1)
            for j in range(i + 1, len(person_heads)):
                second = person_heads[j]
                if j not in active or first["class_name"] == second["class_name"]:
                    continue
                bx1, by1, bx2, by2 = second["xyxy"]
                intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
                area_a = aw * ah
                area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
                iou = intersection / max(1, area_a + area_b - intersection)
                bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
                center_distance = (
                    ((acx - bcx) / max(aw, bx2 - bx1, 1)) ** 2
                    + ((acy - bcy) / max(ah, by2 - by1, 1)) ** 2
                ) ** 0.5
                if iou < 0.30 and not (intersection > 0 and center_distance <= 0.20):
                    continue
                first_score = first["confidence"] + (0.03 if first["second_pass_support"] else 0)
                second_score = second["confidence"] + (0.03 if second["second_pass_support"] else 0)
                if abs(first_score - second_score) <= 0.04:
                    uncertain_heads.extend((first, second))
                    active.discard(i)
                    active.discard(j)
                    break
                loser = j if first_score > second_score else i
                active.discard(loser)
                print(f"REJECTED: {person_heads[loser]['class_name']} conflict -> stronger overlapping class")
                if loser == i:
                    break
        validated_heads.extend(person_heads[index] for index in active)

    validated_by_bike = {index: [] for index in range(len(bikes))}
    for head in validated_heads:
        validated_by_bike[head["bike_index"]].append(head)

    print("VALID:")
    for head in validated_heads:
        print(
            f"{head['class_name'].replace('-', ' ').title()} {head['confidence']:.2f} "
            f"-> Person {head['person_index'] + 1} -> Bike {head['bike_index'] + 1}"
        )

    # Only validated, associated no-helmet occupants can trigger a violation.
    valid_no_helmet = [
        head for head in validated_heads if head["class_name"] == "no-helmet"
    ]
    violation = len(valid_no_helmet) > 0
    valid_plates = final_plate_detections

    # Draw confident bikes and plates plus only validated occupant heads.
    final_detections = bikes + valid_plates + validated_heads
    print("FINAL:")
    for bike_index, bike in enumerate(bikes, start=1):
        occupants = validated_by_bike[bike_index - 1]
        if not occupants:
            print(f"Motorcycle {bike_index} -> no validated occupants")
        for head in occupants:
            class_label = head["class_name"].replace("-", " ").title()
            print(
                f"Motorcycle {bike_index} -> Person {head['person_index'] + 1} "
                f"-> {class_label}"
            )
    if uncertain_heads:
        print(f"Uncertain overlapping head conflicts: {len(uncertain_heads)}")

    # YOLO plate detection is independent of OCR success. Run OCR for every
    # deduplicated plate using several enhanced views and retain its best text.
    # Process every valid plate and order results from the left side of the
    # image to the right for a predictable numbered list.
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
        best_text = "Unreadable"
        if plate_crop.size:
            def registration_candidate(text):
                cleaned = re.sub(r"[^A-Z0-9-]", "", text.upper())
                cleaned = re.sub(r"-+", "-", cleaned).strip("-")
                digits = sum(char.isdigit() for char in cleaned)
                letters = sum(char.isalpha() for char in cleaned)
                if 5 <= len(cleaned) <= 12 and digits >= 3 and 1 <= letters <= 5:
                    return cleaned
                return None

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
                if candidate is None or confidence < 0.15:
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
                    enlarged = cv2.resize(
                        source_crop, None, fx=scale, fy=scale,
                        interpolation=cv2.INTER_CUBIC,
                    )
                    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
                    clahe = cv2.createCLAHE(
                        clipLimit=2.0, tileGridSize=(8, 8)
                    ).apply(gray)
                    blurred = cv2.GaussianBlur(clahe, (0, 0), 2.0)
                    sharpened = cv2.addWeighted(clahe, 1.7, blurred, -0.7, 0)
                    denoised = cv2.fastNlMeansDenoising(
                        clahe, None, h=8, templateWindowSize=7, searchWindowSize=21
                    )
                    denoised_sharpened = cv2.addWeighted(
                        denoised, 1.7,
                        cv2.GaussianBlur(denoised, (0, 0), 2.0), -0.7, 0,
                    )
                    _, otsu = cv2.threshold(
                        clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
                    )
                    adaptive_block_size = min(31, min(clahe.shape))
                    if adaptive_block_size % 2 == 0:
                        adaptive_block_size -= 1
                    adaptive = cv2.adaptiveThreshold(
                        clahe, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                        cv2.THRESH_BINARY, max(3, adaptive_block_size), 11,
                    )
                    views = [
                        ("Original", cv2.cvtColor(enlarged, cv2.COLOR_BGR2RGB)),
                        ("Grayscale", gray),
                        ("CLAHE", clahe),
                        ("Sharpened", sharpened),
                        ("Denoised + sharpened", denoised_sharpened),
                        ("Otsu", otsu),
                        ("Inverted Otsu", cv2.bitwise_not(otsu)),
                        ("Adaptive", adaptive),
                    ]

                    for view_name, view in views:
                        variant = f"{crop_name} {view_name} {scale}x"
                        ocr_items = reader.readtext(
                            view, detail=1, paragraph=False, allowlist=allowlist
                        )
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
                variants = {entry["variant"] for entry in entries}
                mean_confidence = sum(entry["confidence"] for entry in entries) / len(entries)
                mean_position = sum(entry["position"] for entry in entries) / len(entries)
                agreement = min(1.0, len(variants) / 4.0)
                score = 0.50 * mean_confidence + 0.35 * agreement + 0.10 * mean_position
                formats = {}
                for entry in entries:
                    formats.setdefault(entry["text"], []).append(entry["confidence"])
                chosen_format = max(
                    formats,
                    key=lambda text: (
                        len(formats[text]),
                        sum(formats[text]) / len(formats[text]),
                        "-" in text,
                    ),
                )
                ranked_candidates.append(
                    (score, mean_confidence, chosen_format, normalized, len(variants))
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
                best_text = "Unreadable"
        else:
            best_text = "Unreadable"
            print("Selected candidate: Unreadable (empty plate crop)")
        plate_results.append(best_text)
        print(f"PLATE {len(plate_results)} SELECTED: {best_text}")

    # Return every plate in detection order; an unreadable OCR result still
    # represents a successfully detected plate.
    if not plate_detections:
        plate_text = "License Plates Detected: 0\nNot detected"
    else:
        plate_text = "License Plates Detected: {}\n{}".format(
            len(plate_results),
            "\n".join(
                f"{index}. {text} — detection {detection['confidence'] * 100:.0f}%"
                for index, (text, detection) in enumerate(
                    zip(plate_results, plate_detections), 1
                )
            ),
        )

    # Draw only final validated detections with compact percentage labels.
    detected_image = original.copy()
    colors = {
        "bike": (255, 0, 0),          # blue in OpenCV's BGR color order
        "helmet": (0, 190, 0),       # green
        "no-helmet": (0, 0, 220),    # red
        "number-plate": (180, 0, 180),  # purple
    }
    display_names = {
        "bike": "Bike",
        "helmet": "Helmet",
        "no-helmet": "No Helmet",
        "number-plate": "Plate",
    }
    placed_labels = []
    for detection in final_detections:
        x1, y1, x2, y2 = map(int, detection["xyxy"])
        color = colors.get(detection["class_name"], (0, 220, 255))
        cv2.rectangle(detected_image, (x1, y1), (x2, y2), color, 2)
    for detection in sorted(final_detections, key=lambda item: (item["xyxy"][1], item["xyxy"][0])):
        x1, y1, x2, y2 = map(int, detection["xyxy"])
        class_name = detection["class_name"]
        color = colors.get(class_name, (0, 220, 255))
        label = f"{display_names.get(class_name, class_name)} {detection['confidence'] * 100:.0f}%"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1
        )
        label_width = min(width, text_width + 8)
        label_height = text_height + baseline + 6
        label_x = min(max(0, x1), max(0, width - label_width))
        candidate_ys = [y1 - label_height, y2 + 2, y1 + 2]
        candidate_ys.extend(y1 - label_height - step * (label_height + 2) for step in range(1, 5))
        label_y = min(max(0, y1 - label_height), max(0, height - label_height))
        for candidate_y in candidate_ys:
            candidate_y = min(max(0, candidate_y), max(0, height - label_height))
            candidate_rect = (label_x, candidate_y, label_x + label_width, candidate_y + label_height)
            overlaps = any(
                candidate_rect[0] < old[2] and candidate_rect[2] > old[0]
                and candidate_rect[1] < old[3] and candidate_rect[3] > old[1]
                for old in placed_labels
            )
            if not overlaps:
                label_y = candidate_y
                break
        cv2.rectangle(
            detected_image, (label_x, label_y),
            (label_x + label_width, label_y + label_height), color, -1
        )
        cv2.putText(
            detected_image, label, (label_x + 4, label_y + text_height + 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA
        )
        placed_labels.append((label_x, label_y, label_x + label_width, label_y + label_height))
    detected_image = cv2.cvtColor(detected_image, cv2.COLOR_BGR2RGB)

    if violation:
        status = "🚨 HELMET VIOLATION DETECTED"
    else:
        status = "✅ NO VIOLATION DETECTED"

    helmeted_count = sum(head["class_name"] == "helmet" for head in validated_heads)
    no_helmet_count = len(valid_no_helmet)
    detection_summary_lines = [
        f"Motorcycles Detected: {len(bikes)}",
        f"Helmeted Occupants: {helmeted_count}",
        f"No-Helmet Occupants: {no_helmet_count}",
        f"License Plates Detected: {len(plate_detections)}",
        "",
        "Occupants by Motorcycle:",
    ]
    for bike_index, bike in enumerate(bikes, start=1):
        occupant_labels = [
            f"{head['class_name']} {head['confidence'] * 100:.0f}%"
            for head in validated_by_bike[bike_index - 1]
        ]
        detection_summary_lines.append(
            f"Bike {bike_index}: " + (" + ".join(occupant_labels) if occupant_labels else "no associated occupants")
        )
    if uncertain_heads:
        detection_summary_lines.append(f"Uncertain head conflicts excluded: {len(uncertain_heads)}")
    detection_summary = "\n".join(detection_summary_lines)

    return (
        detected_image,
        status,
        plate_text,
        detection_summary
    )


# Web Interface
with gr.Blocks(
    title="Motorcycle Helmet Violation Detection"
) as app:

    gr.Markdown(
        """
        # 🏍️ Motorcycle Helmet Violation Detection

        Deep Learning-based traffic safety system using **YOLOv8 + EasyOCR**.

        Upload a motorcycle traffic image to detect helmets,
        no-helmet violations, motorcycles, and license plates.
        """
    )

    with gr.Row():

        with gr.Column():
            input_image = gr.Image(
                type="numpy",
                label="Upload Traffic Image"
            )

            analyze_button = gr.Button(
                "Analyze Image",
                variant="primary"
            )

            clear_button = gr.ClearButton(
                value="Clear"
            )

        with gr.Column():
            result_image = gr.Image(
                label="Detection Result"
            )

    gr.Markdown("## Detection Summary")

    with gr.Row():

        status_box = gr.Textbox(
            label="Violation Status"
        )

        plate_box = gr.Textbox(
            label="Detected License Plates",
            lines=8,
        )

    detection_box = gr.Textbox(
        label="Detected Objects & Confidence",
        lines=6
    )

    analyze_button.click(
        fn=web_detect,
        inputs=input_image,
        outputs=[
            result_image,
            status_box,
            plate_box,
            detection_box
        ]
    )

    clear_button.add([
        input_image,
        result_image,
        status_box,
        plate_box,
        detection_box
    ])


if __name__ == "__main__":
    app.launch()
