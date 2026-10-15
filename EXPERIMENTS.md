# Experiments

This file records the available baseline/fine-tuned comparison. The values are project-reported; the restructuring work did not train models or rerun evaluation.

## Comparison

| Metric | Baseline | Fine-tuned |
| --- | ---: | ---: |
| Precision | 0.923 | 0.918 |
| Recall | 0.881 | 0.888 |
| mAP@50 | 0.942 | 0.943 |
| mAP@50:95 | 0.669 | 0.691 |
| No-Helmet precision | 0.912 | 0.906 |
| No-Helmet recall | 0.822 | 0.834 |
| No-Helmet mAP@50 | 0.911 | 0.917 |
| No-Helmet mAP@50:95 | 0.605 | 0.621 |

## Reported configuration

- Dataset: 8,481 images; 5,936 train, 1,690 validation, 855 test.
- Detector: YOLOv8; image size 640 × 640; batch size 16; 40 epochs.
- Runtime/training stack: PyTorch + Ultralytics; Google Colab Tesla T4.
- Main objective: helmet/no-helmet detection. Motorcycle and plate detection/OCR support the application.

## Observations

- Overall recall increased from 0.881 to 0.888 and mAP@50:95 increased from 0.669 to 0.691 after fine-tuning.
- No-helmet recall increased from 0.822 to 0.834; no-helmet mAP@50:95 increased from 0.605 to 0.621.
- Precision decreased slightly overall (0.923 to 0.918) and for no-helmet (0.912 to 0.906).
- The reported changes are consistent with the project goal of improving difficult no-helmet cases, but no separate difficult-case test-set metrics are present.

## Recordkeeping limits

Per-run configuration files, raw validation logs, random seeds, and the exact dataset YAML used for these reported metrics are not tracked. The ignored local dataset export is Version 3 and has different split counts. Treat the figures above as supplied project records, not independently reproducible results from this checkout. Future runs should record the dataset version, checkpoint, command/configuration, environment versions, and validation output together.
