# Motorcycle Helmet Violation Detection

A Gradio application that detects motorcycles, helmet use, and license plates in traffic images. EasyOCR reads detected plates.

## Project structure

```text
.
├── app/                 # Gradio application
├── model/
│   ├── coco/            # COCO weights for person and motorcycle validation
│   ├── archive/         # Previous custom model weights
│   └── helmet_best.pt   # Current helmet and plate detector
├── notebook/            # Training notebook
├── results/             # Training plots and sample output
├── requirements.txt
└── README.md
```

Training data is kept locally under `dataset/` and ignored by Git because of its size. The virtual environment and temporary inference images are also local-only.

## Run the app

Install the dependencies and launch Gradio from the repository root:

```bash
pip install -r requirements.txt
python app/app.py
```

Upload an image to view motorcycle, helmet/no-helmet, and license-plate detections, along with the violation status and OCR result.

## Models

The custom detector recognizes `bike`, `helmet`, `no-helmet`, and `number-plate`. The COCO model in `model/coco/` supplies person and motorcycle validation detections. Model files are checked into the repository.

## Dataset

The training dataset is *Helmet and Number Plate Detection for Motorbike Safety – Version 3* from [Roboflow Universe](https://universe.roboflow.com/helmet-and-number-plate-detection-project/helmet-and-number-plate-detection-for-motorbike-safety-iityz). The dataset itself is not tracked in this repository.

## Results

Validation metrics:

- mAP@50: 94.2%
- mAP@50–95: 67.5%
- No-helmet precision: 91.7%
- No-helmet recall: 81.7%

![Sample detection](results/sample_detection.png)

![Training results](results/results.png)

![Confusion matrix](results/confusion_matrix.png)
