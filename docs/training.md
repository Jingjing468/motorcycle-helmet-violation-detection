# Training

## Reported setup

| Setting | Project-reported value |
| --- | --- |
| Detector | YOLOv8 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Epochs | 40 |
| Hardware | Google Colab NVIDIA Tesla T4 |
| Framework | PyTorch + Ultralytics |
| Dataset split | 5,936 train / 1,690 validation / 855 test (8,481 total) |

These settings were supplied in the project information and are documented as reported configuration. This repository update did not launch a training run or independently verify its logs. Detailed optimizer, seed, augmentation, and run-by-run logs are not present, so they are not specified here.

## Checkpoints

The application uses `model/helmet_best.pt` as its fine-tuned custom model. The earlier model is stored at `model/archive/helmet_best_old.pt`. Both files are YOLOv8 custom checkpoints. The COCO support model is `model/coco/yolov8n.pt`. See [`checkpoints/README.md`](../checkpoints/README.md).

The notebook path is `notebooks/helmet_violation_detection.ipynb`; the currently tracked notebook file is empty and does not contain a reproducible training procedure. The training interface and inference model loading are separate: the application loads the existing weights and does not train on startup.

## Reproducing training

To reproduce the experiment, obtain the correct dataset version and YAML configuration, then record the exact Ultralytics command/configuration, software versions, random seed, output directory, and resulting metrics. Keep the full dataset and generated run artifacts out of Git unless they are intentionally curated for release.
