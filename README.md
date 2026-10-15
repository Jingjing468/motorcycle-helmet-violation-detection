# Motorcycle Helmet Violation Detection

An academic deep-learning project for detecting helmet use in motorcycle traffic images. The primary task is classifying rider headwear as **Helmet** or **No Helmet** and flagging validated no-helmet detections. Motorcycle localization and license plate detection with EasyOCR are supporting features.

## Project overview and objective

Manual review of traffic images is time-consuming, especially when riders are small, occluded, or present in crowded scenes. This project evaluates a YOLOv8-based computer-vision workflow that detects helmet-related classes, associates people with motorcycles, and presents annotated images and a violation summary in a Gradio application.

The objective is to assist image review. The application is a prototype and does not make guaranteed or automatic enforcement decisions.

## Features

- Detect **Helmet** and **No Helmet**; this is the main project focus.
- Associate candidate riders with motorcycles and report validated rider status.
- Detect motorcycles and number plates as supporting features.
- Attempt plate text recognition with EasyOCR.
- Run image inference through a local Gradio interface, with annotated output, counts, confidence values, and detection crops.

## Dataset

The reported project dataset contains **8,481 images**: 5,936 training, 1,690 validation, and 855 test images. The four classes are:

| ID | Class | Role |
| ---: | --- | --- |
| 0 | `bike` | Motorcycle support |
| 1 | `helmet` | Main task |
| 2 | `no-helmet` | Main task |
| 3 | `number-plate` | Supporting task |

The full dataset is not included in this repository. Two small example input images are in [`data/samples/`](data/samples/). The ignored local `dataset/` export, if present, is Roboflow Version 3 with different split counts; it must not be treated as the 8,481-image experiment dataset. See [Dataset notes](docs/dataset.md).

## Architecture

The custom YOLOv8 model predicts bike, helmet, no-helmet, and number-plate boxes. A COCO-pretrained YOLO model supplies person/motorcycle detections for rider association. Post-processing validates headwear detections against associated riders; only validated no-helmet occupants trigger a violation. Plate detection and OCR follow a separate supporting path. The Gradio interface is in `app/`; reusable inference modules are in `src/`; paths, classes, and configurable thresholds are in `config/`.

See [Architecture](docs/architecture.md) for the workflow and module map.

## Project structure

```text
.
├── app/                 # Gradio entry point and UI
├── checkpoints/         # Documentation for model checkpoints stored in model/
├── config/              # Class names, paths, settings, thresholds
├── data/                # Dataset notes and small sample inputs; no full dataset
├── docs/                # Architecture, data, training, evaluation, demo notes
├── model/               # Fine-tuned, archived baseline, and COCO weights
├── notebooks/           # Notebook location (current notebook is empty)
├── results/             # Existing example images and figures
├── scripts/             # Validation, evaluation, benchmark, single-image tools
├── slides/              # Presentation asset guidance
├── src/                 # Detection, association, OCR, post-processing, pipeline
├── test_images/         # User-provided inference images (ignored)
├── tests/               # Model-independent unit tests
├── EXPERIMENTS.md
├── PROJECT_CONTEXT.md
├── TASKS.md
└── README.md
```

## Training setup

Project-reported training used YOLOv8 at **640 × 640**, batch size **16**, for **40 epochs**, on a **Google Colab Tesla T4**, with **PyTorch and Ultralytics**. These are recorded project settings, not a training run reproduced during repository organization. See [Training](docs/training.md).

## Evaluation results

The table contains the project-reported baseline and fine-tuned results. They are transcribed from project information and were not recomputed as part of this repository update.

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

See [Evaluation](docs/evaluation.md) and [`EXPERIMENTS.md`](EXPERIMENTS.md) for interpretation and provenance notes.

## Installation

From the repository root, create an environment and install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The app expects the custom and COCO weights at the paths configured in `config/paths.py`. The current fine-tuned weights are in `model/`; see [`checkpoints/README.md`](checkpoints/README.md).

## Run the Gradio application

```bash
python app/app.py
```

Open the local URL printed by Gradio, usually `http://127.0.0.1:7860`. Upload an image, select **Analyze Image**, and use **Clear** to reset. Inference runs locally and can be CPU-intensive. The application accepts images; video/live-camera inference is not implemented.

## Demo workflow

1. Upload a helmeted rider image and inspect the annotated image, helmet crop, rider status, and confidence.
2. Upload a no-helmet rider image and inspect the validated violation result.
3. Review supporting motorcycle and license-plate detections and any EasyOCR output. OCR may report `Unreadable`.

See [Demo guide](docs/demo.md) for presentation notes and interpretation cautions.

## Limitations

- Small or distant heads and partial occlusion can reduce helmet detection accuracy.
- Rear views, crowded scenes, and multiple riders challenge detection and rider association.
- Hats and caps may be confused with helmets.
- Plate detection and OCR are supporting, experimental features; small, angled, blurred, or occluded plates may be unreadable or misread.
- A missing helmet detection is not proof that a rider is unhelmeted; a violation requires a validated `no-helmet` result.

More detail is in [Limitations](docs/limitations.md).

## Future work

Collect more difficult training examples; improve crowded-scene detection and rider association; improve plate detection and OCR; compare another detector using consistent splits; and investigate video/live-camera inference. See [Future Work](docs/future_work.md) and [`TASKS.md`](TASKS.md).

## AI-use disclosure

ChatGPT and OpenAI Codex assisted with debugging, code organization, UI work, and documentation. The project author remains responsible for reviewing the implementation and claims. Reported metrics are project-provided and are not newly generated here. See [AI usage notes](docs/ai_usage.md).

## Author and course

- **Author:** Jing Jing ([GitHub](https://github.com/Jingjing468))
- **Course:** Deep Learning final project; course title and institution were not included in the project materials.

## Tests and utilities

Run the unit tests:

```bash
python -m unittest discover -s tests -v
```

Other tools:

```bash
python scripts/validate_project.py
python scripts/validate_project.py --load-models
python scripts/test_single_image.py --image path/to/image.jpg
python scripts/benchmark_fps.py --images path/to/images/
python scripts/evaluate_model.py --data path/to/data.yaml
```

The evaluation script requires a compatible external dataset YAML and dataset files. For the full project map, dataset/checkpoint notes, and experiment context, see [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).
