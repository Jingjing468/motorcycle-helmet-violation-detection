# Project Tasks

This is a forward-looking checklist based on documented limitations and future-work goals. It does not imply the tasks have already been completed.

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
