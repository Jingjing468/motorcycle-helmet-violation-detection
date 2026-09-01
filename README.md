# Motorcycle Helmet Violation Detection

A Deep Learning-based traffic safety system that detects motorcycle riders, helmets, helmet violations, and license plates from traffic images using YOLOv8 and OCR.

## Project Overview

This project automatically analyzes motorcycle images and detects:

- Motorcycle / Bike
- Helmet
- No Helmet
- Number Plate

When a rider without a helmet is detected, the system marks it as a helmet violation.

The detected license plate is cropped and processed using EasyOCR to extract the plate number.

## Features

- Motorcycle detection
- Helmet detection
- No-helmet violation detection
- License plate detection
- License plate OCR
- Confidence score display
- Violation status
- Evidence image saving
- CSV violation records
- Gradio web interface

## Deep Learning Model

The project uses YOLOv8 for object detection.

Classes:

1. bike
2. helmet
3. no-helmet
4. number-plate

## Dataset

Dataset used:

Helmet and Number Plate Detection for Motorbike Safety - Version 3

Source:
https://universe.roboflow.com/helmet-and-number-plate-detection-project/helmet-and-number-plate-detection-for-motorbike-safety-iityz

The dataset is not included in this repository because of its large size.

## Model Performance

Validation Results:

- mAP@50: 94.2%
- mAP@50-95: 67.5%

Per-class mAP@50-95:

- Bike: 79.8%
- Helmet: 63.1%
- No Helmet: 60.2%
- Number Plate: 66.8%

No-Helmet Detection:

- Precision: 91.7%
- Recall: 81.7%

## System Architecture

Traffic Image

↓

YOLOv8 Object Detection

↓

Bike / Helmet / No-Helmet / Number Plate

↓

Helmet Violation Detection

↓

Number Plate Cropping

↓

Image Enhancement

↓

EasyOCR

↓

License Plate Text

↓

Violation Result + Evidence

## Technologies

- Python
- PyTorch
- YOLOv8
- Ultralytics
- OpenCV
- EasyOCR
- Gradio
- Google Colab

## Project Structure

```text
helmet_violation_detection/
├── app/
│   └── app.py
├── dataset/
├── model/
│   └── helmet_best.pt
├── notebook/
│   └── helmet_violation_detection.ipynb
├── results/
│   ├── confusion_matrix.png
│   ├── results.png
│   └── sample_detection.png
├── test_images/
├── test_videos/
├── .gitignore
├── README.md
└── requirements.txt