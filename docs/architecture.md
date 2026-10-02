# Architecture

The custom detector uses the YOLOv8 family: a convolutional backbone extracts image features, a neck combines features at multiple scales, and a detection head predicts class scores and bounding boxes. This repository does not contain training source that would justify describing a different architecture for the two saved custom checkpoints; the weights use the same detector architecture.

The application loads the custom helmet/plate/bike model and the COCO pretrained model once. The custom model finds helmet classes, candidate motorcycles, and plates. COCO person and motorcycle detections support rider validation and bike localization. The pipeline associates people with motorcycles, validates head predictions, optionally runs plate OCR, draws public detections, and returns the violation status and summary. Internal person boxes are not drawn.

The Gradio boundary is in `app/`; inference and reusable helpers are in `src/`; paths and retained numeric thresholds are in `config/`.
