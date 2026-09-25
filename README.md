# Motorcycle Helmet Violation Detection

An image-based computer vision project for identifying motorcycles, helmet use, and license plates. The application combines a custom YOLOv8 detector, a supporting COCO-pretrained YOLO model, OpenCV image processing, and EasyOCR in a local Gradio interface.

## 1. Project Overview

Motorcycle Helmet Violation Detection analyzes a traffic image and locates motorcycles, helmets, people without helmets, and number plates. It associates head detections with people and motorcycles before deciding whether the image contains a validated no-helmet case. Detected plate regions are also passed to optical character recognition (OCR) to try to read their registration text.

The project includes a Gradio web interface for uploading an image and viewing the annotated result, violation status, plate reading, and detection summary. It is an assistive prototype; its output should be reviewed by a person.

## 2. Problem Statement

Motorcycle riders may travel without helmets, and reviewing traffic images manually can take considerable effort. Identifying a rider in a crowded scene, deciding whether the rider is wearing a helmet, and reading a small or partially visible plate can each be difficult. Deep learning can help automate parts of this review by locating relevant objects and extracting candidate plate text.

This project explores that workflow. It is not intended to replace traffic officers or make enforcement decisions without human review.

## 3. Proposed Solution

The system processes one uploaded image through object detection, association, OCR, and a rule-based violation decision:

```text
Input image
    → motorcycle and person detection
    → helmet / no-helmet detection
    → number-plate detection
    → plate crop and OCR
    → validated violation decision
    → annotated image and summary in Gradio
```

The displayed custom-model labels are **Bike**, **Helmet**, **No Helmet**, and **Plate**. If at least one validated motorcycle occupant is classified as no-helmet, the application reports a helmet violation.

## 4. Main Features

- Detects motorcycles, helmets, no-helmet cases, and number plates.
- Associates people and head detections with detected motorcycles to support multiple occupants.
- Uses a second inference pass on enlarged motorcycle or person crops to examine small heads.
- Detects and processes multiple plates in an image.
- Applies several image variants before OCR and reports `Unreadable` when no candidate meets the implemented criteria.
- Draws colored bounding boxes and confidence percentages on the result image.
- Shows violation status, plate count and OCR text, and an occupant summary.
- Runs inference locally with the fine-tuned model files included in the repository.
- Provides an interactive Gradio interface.

## 5. Technologies Used

| Technology | Use in this project |
| --- | --- |
| Python | Application and inference pipeline. |
| PyTorch | Deep learning framework used by the Ultralytics model runtime. |
| Ultralytics YOLOv8 | Custom object detection and the supporting COCO-pretrained model. |
| EasyOCR | Reads candidate registration text from detected plate crops. |
| OpenCV | Image conversion, resizing, crop processing, contrast enhancement, thresholding, and drawing result boxes. |
| Gradio | Local web interface for image upload and display of results. |
| Google Colab | Reported environment for model training and fine-tuning. |
| Roboflow | Source and export format for the public helmet and number-plate dataset. |
| Git and GitHub | Source control and project hosting. |

## 6. Dataset

The project uses the Roboflow Universe dataset [Helmet and Number Plate Detection for Motorbike Safety](https://universe.roboflow.com/helmet-and-number-plate-detection-project/helmet-and-number-plate-detection-for-motorbike-safety-iityz). The custom detector uses these class IDs:

| ID | Class |
| ---: | --- |
| 0 | bike |
| 1 | helmet |
| 2 | no-helmet |
| 3 | number-plate |

Project training notes identify a Version 4 dataset of approximately **8,481 images**, with additional difficult no-helmet examples such as rear views, crowded scenes, and small heads. These examples were added to improve coverage of cases that were challenging for the earlier model.

There is a dataset-version discrepancy in the local repository snapshot: the ignored `dataset/` directory contains a Roboflow **Version 3** YOLOv8 export. Its metadata reports 20,287 images, and the local split directories contain 17,742 training, 1,690 validation, and 855 test images. Those counts describe the local Version 3 export, not the Version 4 training set described in the project notes. The dataset is ignored by Git and is not included in the repository clone.

## 7. Data Preparation

The Roboflow export uses YOLO bounding-box labels and defines separate training, validation, and test directories. Its Version 3 metadata records auto-orientation, resizing to fit within 640 × 640 pixels, and generated variants using random rotation between −15° and +15° and Gaussian blur up to 1.5 pixels.

The project notes describe adding difficult no-helmet examples for fine-tuning. The exact Version 4 split counts, the exact preparation applied to those additional examples, and the augmentation settings for the final training run are not recorded in the repository. The local training notebook file is empty, so this README does not infer those missing details from it.

## 8. Model Architecture

YOLOv8 is a one-stage object detector: it predicts object bounding boxes, class labels, and confidence scores in a single detection pipeline. This project uses a fine-tuned YOLOv8 model for four custom classes: `bike`, `helmet`, `no-helmet`, and `number-plate`.

The app also loads `model/coco/yolov8n.pt`, a COCO-pretrained YOLOv8 nano model. It supplies person and motorcycle detections that help validate custom head detections and associate people with bikes. The custom and supporting models have different weights and roles, but they are both YOLOv8 models; they are not two distinct architecture comparisons.

## 9. Training Process

According to the project training notes, training and fine-tuning were performed in Google Colab with PyTorch and Ultralytics on an NVIDIA Tesla T4 GPU. An initial custom detector was trained first. A later experiment added difficult no-helmet images and continued training from the earlier model for 10 additional epochs rather than starting from random weights. Continuing training from an existing model with additional or improved examples is called **fine-tuning**.

The repository does not contain a populated training notebook or the complete training configuration, so details such as batch size, optimizer settings, and exact split used for each reported run are not documented here.

## 10. Model Evaluation

The following values are the project-reported evaluation results:

| Metric | Original model | Fine-tuned model |
| --- | ---: | ---: |
| Precision | — | 0.918 |
| Recall | — | 0.889 |
| mAP@50 | 0.942 | 0.942 |
| mAP@50:95 | 0.675 | 0.690 |

### Per-class mAP@50:95

| Class | Original model | Fine-tuned model |
| --- | ---: | ---: |
| bike | 0.798 | 0.801 |
| helmet | 0.631 | 0.643 |
| no-helmet | 0.602 | 0.622 |
| number-plate | 0.668 | 0.694 |

The reported overall mAP@50:95 increased from 0.675 to 0.690. No-helmet mAP@50:95 increased from 0.602 to 0.622, and no-helmet recall increased from 0.817 to 0.836. These class-specific values are distinct from the fine-tuned model's reported overall recall of 0.889.

In simple terms:

- **Precision** measures how often the model's positive detections are correct.
- **Recall** measures how many of the relevant objects the model successfully finds.
- **mAP@50** is mean average precision when a predicted box counts as a match at intersection-over-union (IoU) of 0.50.
- **mAP@50:95** averages mean average precision across IoU thresholds from 0.50 through 0.95, so it also reflects how closely predicted boxes align with the labelled objects.

## 11. Improvement Experiment

The fine-tuning experiment focused on difficult no-helmet examples, including rear-view riders, crowded scenes, and small heads. Project notes report 10 additional training epochs and this example comparison:

| Example image detections | Original model | Fine-tuned model |
| --- | ---: | ---: |
| Bikes | 5 | 5 |
| Helmets | 12 | 9 |
| No-helmets | 1 | 4 |

This example indicates that the fine-tuned model identified more no-helmet cases in that image. It is one example and does not establish perfect accuracy or guarantee the same improvement on every scene.

## 12. System Workflow

```text
User uploads image
        ↓
Custom YOLO and COCO YOLO inference
        ↓
Bike / person association
        ↓
Helmet / no-helmet detection and validation
        ↓
Number-plate detection
        ↓
Plate crop and image enhancement
        ↓
EasyOCR text recognition
        ↓
Violation decision
        ↓
Annotated image and summary in Gradio
```

## 13. License Plate Detection and OCR

Plate detection and text recognition are separate steps. The custom YOLO model first locates a `number-plate` region. The application expands and crops that region from the original image, enlarges it, and creates several views for OCR, including color, grayscale, CLAHE-enhanced, sharpened, and thresholded versions. EasyOCR reads text from these candidate views.

The code filters OCR output to ASCII letters, digits, and hyphens, then accepts registration-like candidates within its implemented length and character criteria. This removes non-ASCII text such as Khmer province headings from the registration candidate. It does not translate or recognize the Khmer text. Multiple plates are processed independently and listed in the result. A plate can be detected even when its text is unreadable; in that case, the app reports `Unreadable` for the OCR result.

## 14. Helmet Violation Logic

The app validates head detections through spatial association:

1. The COCO model detects people and motorcycles; the custom model detects bikes and helmet classes.
2. A helmet or no-helmet box must fall in the upper portion of a detected person.
3. The person must be associated with a detected motorcycle using their relative position.
4. A custom `no-helmet` detection is required to report a violation.

The absence of a helmet box alone does not count as a violation. Person and motorcycle association, confidence thresholds, and additional crop-based detections are used to reduce unsupported head detections. These checks can reduce false positives but do not eliminate them.

## 15. Web Application

The Gradio interface lets a user upload an image and click **Analyze Image**. It displays the annotated image, violation status, detected plate list and OCR text, and a summary of bikes and associated occupants. The app runs locally and uses the model files in the repository; Google Colab is not required for inference after dependencies and model files are available.

## 16. Project Structure

```text
motorcycle-helmet-violation-detection/
├── app/
│   └── app.py
├── model/
│   ├── archive/
│   │   └── helmet_best_old.pt
│   ├── coco/
│   │   └── yolov8n.pt
│   └── helmet_best.pt
├── notebook/
│   └── helmet_violation_detection.ipynb
├── results/
│   ├── confusion_matrix.png
│   ├── results.png
│   └── sample_detection.png
├── requirements.txt
├── .gitignore
└── README.md
```

The local `dataset/`, `.venv/`, and temporary inference images are excluded from Git. The `test_images/` and `test_videos/` directories are currently empty and are not tracked.

## 17. Installation

Clone the repository and enter its directory:

```bash
git clone https://github.com/Jingjing468/motorcycle-helmet-violation-detection.git
cd motorcycle-helmet-violation-detection
```

Create and activate a Python 3.11 virtual environment on macOS or Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

## 18. Run the Application

From the repository root, run:

```bash
python app/app.py
```

Open the local Gradio URL printed in the terminal. It is commonly `http://127.0.0.1:7860`.

## 19. Hardware and Environment

- **Training and fine-tuning:** Google Colab, NVIDIA Tesla T4 (as reported in the project training notes).
- **Local testing:** MacBook.
- **Python:** 3.11.
- **Inference speed:** Real-time FPS varies depending on hardware, image size, OCR, and the number of second-pass detections. A formal FPS benchmark is not included in this project.

## 20. Challenges

The project addresses or encounters several difficult conditions: confusing helmet and no-helmet appearances, crowded traffic, small or distant heads, rear-view riders, multiple passengers, overlapping people, low-confidence plate detections, different plate styles, OCR errors, partial plates, and false detections on background objects.

## 21. Error Analysis

Potential failure cases include a bare head classified as a helmet, a helmet classified as no-helmet, a missed plate, a detected plate whose text is unreadable, missed small or distant objects, and overlapping riders that are difficult to associate with a motorcycle. Likely contributing factors include limited examples of difficult cases, class imbalance, small object size, occlusion, lighting, viewpoint, and variation in plate appearance. These are qualitative error categories; the repository does not include a complete per-case error analysis or a quantified breakdown.

## 22. Limitations

- The system is not 100% accurate and its detections need human review.
- Crowded scenes, occlusion, small heads, and multiple-rider association remain challenging.
- OCR quality depends on plate visibility, resolution, lighting, and character style.
- Plate appearance varies across countries and regions; generalization to every style has not been established.
- The current interface analyzes uploaded images; video and live-camera inference are not implemented.
- Inference time depends on hardware and includes multiple detection passes and OCR.
- The repository does not include the training dataset or a populated training notebook, limiting independent reproduction of the reported training runs.

## 23. Future Work

- Collect and label more Cambodian motorcycle and plate images, with appropriate privacy and permissions.
- Improve full-plate annotations and add more difficult no-helmet examples.
- Improve OCR for local plate formats and validate it on a larger, representative sample.
- Improve association for multiple riders and passengers.
- Add video or live-camera support and measure inference speed on defined hardware.
- Evaluate a second, distinct deep learning detector and compare it with YOLOv8 using the same data split and evaluation procedure.
- Explore deployment options after evaluating runtime, accuracy, and operational requirements.
- Test the system on more real Cambodian road scenes.

## 24. Deep Learning Project Requirements

This project demonstrates object detection, transfer learning/fine-tuning, use of PyTorch, evaluation with precision, recall, and mAP, and qualitative error analysis. The original and fine-tuned custom weights are versions of the same YOLOv8 detector architecture. The COCO-pretrained supporting model is also YOLOv8. Therefore, the current repository does **not** demonstrate a comparison of two distinct deep learning approaches. A second architecture must be implemented and evaluated before that comparison requirement can be claimed as complete.

## 25. Results and Screenshots

The project reports the metrics in [Model Evaluation](#10-model-evaluation). Three result image paths are present in `results/`, but the files in the current repository snapshot are empty (0 bytes), so they cannot be displayed here. Replace them with the exported figures before using this README as a final visual report.

- `results/results.png` — training results plot
- `results/confusion_matrix.png` — confusion matrix
- `results/sample_detection.png` — sample detection

## 26. AI Assistance Disclosure

**Tools used:** ChatGPT and OpenAI Codex.

**Scope of use:** Assistance with debugging, code organization, inference-pipeline improvements, technical explanations, and drafting this README.

AI tools did not automatically train the model. The student is responsible for reviewing the code, reported results, and AI-generated suggestions before submission and for confirming that the described behavior matches the final project.

## 27. Author

- **Author:** Yean Sreymom
- **Program:** Bachelor of Software Engineering
- **Institution:** Kirirom Institute of Technology
- **Academic year:** 2026–2027

## 28. References

- [Ultralytics YOLO documentation](https://docs.ultralytics.com/)
- [PyTorch documentation](https://pytorch.org/docs/stable/index.html)
- [EasyOCR project](https://github.com/JaidedAI/EasyOCR)
- [Gradio documentation](https://www.gradio.app/docs)
- [Roboflow Universe dataset: Helmet and Number Plate Detection for Motorbike Safety](https://universe.roboflow.com/helmet-and-number-plate-detection-project/helmet-and-number-plate-detection-for-motorbike-safety-iityz)
