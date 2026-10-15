# Project Context

## Identity and objective

- **Project:** Motorcycle Helmet Violation Detection.
- **Primary focus:** classify associated motorcycle occupants as Helmet or No Helmet and identify validated no-helmet cases.
- **Supporting features:** motorcycle detection; number-plate detection and EasyOCR.
- **Author:** Jing Jing (`Jingjing468`).
- **Course context:** Deep Learning final project; institution and formal course title were not present in the supplied project materials.

## Data and model

- Project-reported dataset: 8,481 images (5,936 train, 1,690 validation, 855 test).
- Classes: `0 bike`, `1 helmet`, `2 no-helmet`, `3 number-plate`.
- Reported training: YOLOv8, 640 × 640, batch 16, 40 epochs, Google Colab Tesla T4, PyTorch + Ultralytics.
- Fine-tuned application checkpoint: `model/helmet_best.pt`.
- Earlier/baseline checkpoint: `model/archive/helmet_best_old.pt`.
- COCO support checkpoint: `model/coco/yolov8n.pt`.

The full 8,481-image dataset is not included. An ignored local Version 3 export may exist under `dataset/`; do not confuse it with the reported experiment dataset or check it into Git.

## Repository conventions

- Start the web app with `python app/app.py` from the repository root.
- Model paths live in `config/paths.py`; class mapping and thresholds live in `config/`.
- Inference orchestration is in `src/core/pipeline.py`; UI code is in `app/`.
- Keep model behavior and OCR behavior separate from documentation/structure changes.
- Store evaluation and demo artifacts under `results/`; presentation-specific materials belong in `slides/`.
- Checkpoint documentation belongs in `checkpoints/`; weight files remain in `model/` to prevent duplication.

## Known project limits

Small/distant heads, occlusion, rear views, crowded scenes, multiple riders, hat/cap confusion, imperfect rider association, and plate OCR quality are known concerns. Video/live-camera inference has not been implemented. See [`docs/limitations.md`](docs/limitations.md) and [`docs/future_work.md`](docs/future_work.md).

## Key documents

- [`README.md`](README.md): repository landing page and run instructions.
- [`EXPERIMENTS.md`](EXPERIMENTS.md): available metric comparison and provenance.
- [`docs/architecture.md`](docs/architecture.md), [`docs/dataset.md`](docs/dataset.md), [`docs/training.md`](docs/training.md), [`docs/evaluation.md`](docs/evaluation.md), [`docs/demo.md`](docs/demo.md).
- [`TASKS.md`](TASKS.md): future development checklist.
