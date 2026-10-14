"""Image inference pipeline extracted from the original Gradio application."""
import cv2  # type: ignore[import-not-found]
import numpy as np

from src.visualization.detection_summary import render_detection_summary
from src.models.helmet_model import predict_full_image
from src.models.coco_model import predict_people_and_motorcycles
from app.ui_helpers import format_violation_status, format_plate_results
from src.visualization.draw_boxes import draw_detections
from src.postprocess.nms import class_aware_nms
from src.postprocess.duplicate_removal import motorcycle_boxes_duplicate as is_duplicate_motorcycle
from src.postprocess.helmet_conflict import select_head_class
from src.association.spatial_matching import person_bike_distance as spatial_person_bike_distance
from src.association.spatial_matching import head_center_in_person_region as spatial_head_center
from config.thresholds import (FINAL_CONFIDENCE, INFERENCE_CONFIDENCE,
                               INFERENCE_IMAGE_SIZE, INFERENCE_IOU, NMS_IOU,
                               PLATE_PASS_CONFIDENCE, SECOND_PASS_HEAD_CONFIDENCE,
                               PERSON_HEAD_CONFIDENCE, CUSTOM_CANDIDATE_CONFIDENCE,
                               COCO_INFERENCE_CONFIDENCE, PERSON_CONFIDENCE,
                               COCO_MOTORCYCLE_CONFIDENCE, SECOND_PASS_RAW_CONFIDENCE,
                               HEAD_RAW_CONFIDENCE, HEAD_RAW_IOU,
                               OCR_FALLBACK_CONFIDENCE)
from src.detection.helmet_detector import head_candidates
from src.detection.motorcycle_detector import eligible_motorcycle_candidates
from src.detection.person_detector import eligible_person_candidates
from src.detection.plate_detector import is_plate_shaped, near_motorcycle
from src.detection.second_pass import upscale_crop, map_scaled_box
from src.association.rider_bike_association import associate_people_to_bikes
from src.ocr.plate_recognizer import recognize_plates
from src.ocr.easyocr_reader import read_plate


def run_pipeline(image, model, coco_model, reader):
    if image is None:
        return None, "Please upload an image", "Not detected", "", []

    if model is None:
        return None, "Model dependency is missing. Please install ultralytics.", "Not detected", "", []

    if reader is None:
        return None, "OCR dependency is missing. Please install easyocr.", "Not detected", "", []

    original = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    height, width = original.shape[:2]

    # Run the normal high-resolution detector on the full image first.
    # A low initial confidence retains candidates; class thresholds below
    # decide which detections are accepted.
    prediction_options = {"conf": INFERENCE_CONFIDENCE, "imgsz": INFERENCE_IMAGE_SIZE,
                          "iou": INFERENCE_IOU, "verbose": False}
    full_results = predict_full_image(model, original)
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
    final_confidence_thresholds = FINAL_CONFIDENCE

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
            if confidence >= CUSTOM_CANDIDATE_CONFIDENCE:
                candidates.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "xyxy": tuple(map(float, box.xyxy[0])),
                    "source": "full-image",
                })
    candidates.extend(
        detection for detection in crop_results
        if detection["confidence"] >= CUSTOM_CANDIDATE_CONFIDENCE
    )
    full_image_head_candidates = [
        detection for detection in head_candidates(candidates)
        if detection["source"] == "full-image"
    ]

    def merge_duplicate_detections(items):
        """Apply class-aware NMS to merge boxes from full and crop passes."""
        return class_aware_nms(items, NMS_IOU)

    def motorcycle_boxes_duplicate(first, second):
        """Match duplicate motorcycle boxes while protecting nearby bikes."""
        return is_duplicate_motorcycle(first, second)

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
        coco_options = {"conf": COCO_INFERENCE_CONFIDENCE, "imgsz": INFERENCE_IMAGE_SIZE,
                        "iou": INFERENCE_IOU, "classes": [0, 3], "verbose": False}
        coco_results = predict_people_and_motorcycles(coco_model, original)
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
                if class_name == "person" and confidence >= PERSON_CONFIDENCE:
                    raw_person_candidates.append(candidate)
                elif class_name == "motorcycle" and confidence >= COCO_MOTORCYCLE_CONFIDENCE:
                    coco_motorcycle_candidates.append(candidate)

        persons = merge_duplicate_detections(eligible_person_candidates(raw_person_candidates))

    # Merge custom bike boxes and COCO motorcycle boxes class-wise. COCO can
    # contribute a bike even when the custom model missed it.
    raw_bike_candidates.extend(coco_motorcycle_candidates)
    custom_bikes = eligible_motorcycle_candidates(raw_bike_candidates)
    bikes = merge_motorcycle_detections(custom_bikes + coco_motorcycle_candidates)
    print(f"FINAL BIKE COUNT: {len(bikes)}")
    raw_detections.extend(bikes)

    # Dedicated full-image plate inference runs below the general detection
    # floor, before app-level confidence or geometry filtering.
    plate_prediction_options = {"imgsz": INFERENCE_IMAGE_SIZE, "conf": PLATE_PASS_CONFIDENCE,
                                "iou": INFERENCE_IOU, "verbose": False}
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

    # Search overlapping lower motorcycle regions so rear and side-mounted
    # plates remain visible even when the bike box is wide or angled.
    motorcycle_plate_candidates = []
    for bike_index, bike in enumerate(bikes, start=1):
        bx1, by1, bx2, by2 = bike["xyxy"]
        bike_w, bike_h = max(1, bx2 - bx1), max(1, by2 - by1)
        search_y1 = max(0, int(by1 + 0.32 * bike_h))
        search_y2 = min(height, int(by2 + 0.12 * bike_h))
        regions = [
            (bx1 - 0.18 * bike_w, bx2 + 0.18 * bike_w, "lower-wide"),
            (bx1 - 0.12 * bike_w, bx1 + 0.64 * bike_w, "left-rear-side"),
            (bx1 + 0.36 * bike_w, bx2 + 0.12 * bike_w, "right-rear-side"),
        ]
        for region_x1, region_x2, region_name in regions:
            crop_x1 = max(0, int(region_x1))
            crop_x2 = min(width, int(region_x2))
            plate_crop = original[search_y1:search_y2, crop_x1:crop_x2]
            if plate_crop.size == 0:
                continue
            print(f"BIKE PLATE CROP {bike_index} ({region_name}):")
            for result in model.predict(source=plate_crop, **plate_prediction_options):
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    if str(model.names[class_id]) != "number-plate":
                        continue
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    mapped_box = (x1 + crop_x1, y1 + search_y1,
                                  x2 + crop_x1, y2 + search_y1)
                    motorcycle_plate_candidates.append({
                        "class_id": class_id, "class_name": "number-plate",
                        "confidence": confidence, "xyxy": mapped_box,
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
        plate_shaped = is_plate_shaped((x1, y1, x2, y2))
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        near_bike = near_motorcycle((x1, y1, x2, y2), bikes)
        if plate["confidence"] >= final_confidence_thresholds["number-plate"]:
            valid_plate_candidates.append(plate)
        elif plate["confidence"] >= PLATE_PASS_CONFIDENCE and near_bike and plate_shaped:
            valid_plate_candidates.append(plate)

    print(f"PLATE CANDIDATE COUNTS: full={len(full_plate_candidates)}, bike_crop={len(motorcycle_plate_candidates)}")
    final_plate_detections = merge_duplicate_detections(valid_plate_candidates)
    fallback_ocr_count = 0
    if not final_plate_detections and bikes:
        def fallback_registration_candidate(text):
            return registration_candidate(text)


        fallback_candidates = []
        for bike_index, bike in enumerate(bikes, start=1):
            bx1, by1, bx2, by2 = bike["xyxy"]
            bike_w, bike_h = max(1, bx2 - bx1), max(1, by2 - by1)
            y1, y2 = max(0, int(by1 + .32 * bike_h)), min(height, int(by2 + .12 * bike_h))
            for rx1, rx2 in ((bx1 - .18 * bike_w, bx2 + .18 * bike_w),
                             (bx1 - .12 * bike_w, bx1 + .64 * bike_w),
                             (bx1 + .36 * bike_w, bx2 + .12 * bike_w)):
                x1, x2 = max(0, int(rx1)), min(width, int(rx2))
                crop = original[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                try:
                    ocr_items = read_plate(reader, cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                except Exception as exc:
                    print(f"Fallback OCR error on bike {bike_index}: {exc}")
                    continue
                for polygon, text, confidence in ocr_items:
                    registration = fallback_registration_candidate(text)
                    if not registration or float(confidence) < OCR_FALLBACK_CONFIDENCE:
                        continue
                    points = np.asarray(polygon, dtype=float)
                    px1, py1 = points.min(axis=0)
                    px2, py2 = points.max(axis=0)
                    pw, ph = px2 - px1, py2 - py1
                    if ph <= 0 or not 1.1 <= pw / ph <= 8.0:
                        continue
                    model_names = model.names
                    plate_class_id = next(
                        (int(i) for i, name in (model_names.items() if isinstance(model_names, dict)
                                                else enumerate(model_names))
                         if str(name) == "number-plate"), -1
                    )
                    fallback_candidates.append({
                        "class_id": plate_class_id,
                        "class_name": "number-plate", "confidence": 0.0,
                        "xyxy": (max(0, x1 + px1 - .08 * pw), max(0, y1 + py1 - .15 * ph),
                                 min(width, x1 + px2 + .08 * pw), min(height, y1 + py2 + .15 * ph)),
                        "source": "ocr-plate-fallback", "plate_text": registration,
                        "ocr_confidence": float(confidence),
                    })
        if fallback_candidates:
            final_plate_detections = merge_duplicate_detections(fallback_candidates)
            fallback_ocr_count = len(final_plate_detections)
    print(f"OCR PLATE FALLBACK COUNT: {fallback_ocr_count}")
    print("FINAL PLATES:")
    for plate_index, plate in enumerate(final_plate_detections, start=1):
        print(f"Plate {plate_index} {plate['confidence']:.2f}")
    raw_detections = [
        d for d in raw_detections if d["class_name"] != "number-plate"
    ] + final_plate_detections

    def person_bike_distance(person, bike):
        """Measure the existing lower-body proximity rule."""
        return spatial_person_bike_distance(person["xyxy"], bike["xyxy"])


    # Inspect an enlarged crop around each detected motorcycle and its upper
    # occupant area. This bike-specific pass focuses on small helmet labels.
    occupant_head_candidates = []
    second_pass_thresholds = {"helmet": SECOND_PASS_HEAD_CONFIDENCE,
                              "no-helmet": SECOND_PASS_HEAD_CONFIDENCE}
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
        enlarged_crop = upscale_crop(occupant_crop, scale)
        print(f"RAW SECOND PASS (bike {bike_index}):")
        for result in model.predict(
            source=enlarged_crop, conf=SECOND_PASS_RAW_CONFIDENCE,
            imgsz=INFERENCE_IMAGE_SIZE, iou=INFERENCE_IOU, verbose=False
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
                    "xyxy": map_scaled_box((x1, y1, x2, y2), crop_x1, crop_y1, scale),
                    "source": "second-pass",
                })

    # For each person associated with a motorcycle, run a focused head crop
    # only when full-image inference found neither head class for that person.
    associated_people = [
        (person_index, person, bike_index)
        for person_index, person, bike_index, _distance
        in associate_people_to_bikes(persons, bikes)
    ]

    def head_center_in_person_region(head, person):
        return spatial_head_center(head["xyxy"], person["xyxy"])


    person_head_thresholds = {"helmet": PERSON_HEAD_CONFIDENCE,
                              "no-helmet": PERSON_HEAD_CONFIDENCE}
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
        px1, py1, px2, py2 = person["xyxy"]
        person_w, person_h = max(1, px2 - px1), max(1, py2 - py1)
        head_x1 = max(0, int(px1 - 0.20 * person_w))
        head_x2 = min(width, int(px2 + 0.20 * person_w))
        head_y1 = max(0, int(py1 - 0.18 * person_h))
        head_y2 = min(height, int(py1 + 0.36 * person_h))
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
            source=enlarged_head_crop, imgsz=960, conf=HEAD_RAW_CONFIDENCE,
            iou=HEAD_RAW_IOU, verbose=False
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

    # Keep every head candidate until it is assigned to a person. Generic NMS
    # can erase a nearby passenger's head before per-person comparison.
    raw_head_detections = [d for d in raw_detections
                           if d["class_name"] in ("helmet", "no-helmet")]
    raw_head_detections.extend(occupant_head_candidates)
    print("RAW:")
    for head in raw_head_detections:
        print(f"{head['class_name'].replace('-', ' ').title()} {head['confidence']:.2f}")

    def find_person_for_head(head):
        """Associate a prediction with a padded upper-head region."""
        hx1, hy1, hx2, hy2 = head["xyxy"]
        head_cx, head_cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
        matches = []
        for person_index, person in enumerate(persons):
            if head_center_in_person_region(head, person):
                px1, py1, px2, py2 = person["xyxy"]
                person_h = max(1, py2 - py1)
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

    # Validate geometry and associate all candidate scores with their rider.
    validated_head_candidates = []
    for head in raw_head_detections:
        person_index = head.get("person_index")
        if person_index is None or person_index >= len(persons) or not head_center_in_person_region(head, persons[person_index]):
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
        hx1, hy1, hx2, hy2 = head["xyxy"]
        shape_ratio = max(hx2 - hx1, hy2 - hy1) / max(1, min(hx2 - hx1, hy2 - hy1))
        if head["class_name"] == "helmet" and shape_ratio > 2.2:
            print(f"REJECTED: {label} {head['confidence']:.2f} -> implausible head-box shape")
            continue
        validated_head_candidates.append({
            **head,
            "person_index": person_index,
            "person_confidence": person["confidence"],
            "bike_index": bike_index,
            "bike_distance": bike_distance,
        })

    # One final classification per associated rider: compare the best spatially
    # valid helmet and no-helmet confidence, with stricter helmet evidence.
    validated_heads = []
    uncertain_heads = []
    for person_index in dict.fromkeys(h["person_index"] for h in validated_head_candidates):
        person_heads = [h for h in validated_head_candidates if h["person_index"] == person_index]
        selected_head = select_head_class(person_heads)
        if selected_head:
            validated_heads.append(selected_head)
        else:
            print(f"PERSON {person_index + 1} FINAL: unknown (no confident class)")


    validated_by_bike = {index: [] for index in range(len(bikes))}
    for head in validated_heads:
        validated_by_bike[head["bike_index"]].append(head)

    print("VALID:")
    final_by_person = {head["person_index"]: head for head in validated_heads}
    for person_index, _person, _bike_index in associated_people:
        final_head = final_by_person.get(person_index)
        if final_head:
            print(f"PERSON {person_index + 1} FINAL: {final_head['class_name']} {final_head['confidence']:.2f}")
        else:
            print(f"PERSON {person_index + 1} FINAL: unknown")
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
    plate_detections, plate_results = recognize_plates(original, valid_plates, reader)

    # Return every plate in detection order; an unreadable OCR result still
    # represents a successfully detected plate.
    plate_text = format_plate_results(plate_detections, plate_results)

    detected_image = draw_detections(original, final_detections)

    status = format_violation_status(violation)

    helmeted_count = sum(head["class_name"] == "helmet" for head in validated_heads)
    no_helmet_count = len(valid_no_helmet)
    detection_summary = render_detection_summary(
        bikes, validated_by_bike, helmeted_count, no_helmet_count, len(plate_detections)
    )

    # Presentation-only crops from the same validated boxes used above; this
    # does not invoke either detector again.
    preview_detections = sorted(
        final_detections,
        key=lambda item: (item["class_name"] not in ("helmet", "no-helmet"), -item["confidence"]),
    )[:12]
    object_gallery = []
    for detection in preview_detections:
        x1, y1, x2, y2 = map(int, detection["xyxy"])
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        crop = original[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        class_label = {
            "bike": "Motorcycle",
            "number-plate": "License Plate",
            "no-helmet": "No Helmet",
        }.get(detection["class_name"], detection["class_name"].title())
        caption = f"{class_label} · {detection['confidence'] * 100:.0f}%"
        object_gallery.append((crop_rgb, caption))

    return (
        detected_image,
        status,
        plate_text,
        detection_summary,
        object_gallery,
    )
