# Project Tasks

This list separates completed project work from remaining goals. Completed items reflect the existing application, checkpoints, reported experiment results, repository assets, and runnable code in this checkout.

## Completed

- [x] Record the four-class YOLO dataset mapping and reported train/validation/test counts.
- [x] Train a YOLOv8 baseline and fine-tune with difficult no-helmet cases; retain the baseline and fine-tuned checkpoints.
- [x] Record the baseline-versus-fine-tuned evaluation comparison.
- [x] Build the working Gradio image application with detection summary and object preview.
- [x] Add supporting motorcycle and number-plate detection with EasyOCR.
- [x] Prepare presentation assets and example images for the final project demo.

## Future work

## Helmet detection

- [ ] Add more difficult helmet/no-helmet examples, including small heads, rear views, occlusion, crowded scenes, multiple riders, and hats/caps.
- [ ] Re-evaluate helmet and no-helmet performance on a versioned, held-out dataset.
- [ ] Review false positives and false negatives and document representative examples.

## Association and motorcycle detection

- [ ] Improve rider-to-motorcycle association in crowded and overlapping scenes.
- [ ] Evaluate motorcycle localization and duplicate handling against labeled examples.

## Plate detection and OCR

- [ ] Improve plate detection for small, angled, blurred, and partially occluded plates.
- [ ] Evaluate OCR on a labeled plate set, including relevant local plate formats.
- [ ] Record OCR accuracy separately from object-detection metrics.

## Evaluation and deployment

- [ ] Compare another detector using the same documented dataset splits and metrics.
- [ ] Preserve run configs, seeds, environment versions, and raw evaluation outputs for future experiments.
- [ ] Measure end-to-end speed on specified hardware.
- [ ] Investigate video/live-camera inference after still-image performance and association are improved.
