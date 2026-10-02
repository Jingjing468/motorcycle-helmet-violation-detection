# Methodology

The project uses annotated traffic images with four custom classes: `bike`, `helmet`, `no-helmet`, and `number-plate`. The custom YOLO detector produces candidate boxes. Application-side spatial rules associate riders with motorcycles and compare the best helmet and no-helmet evidence for each associated person. A positive violation requires a validated no-helmet result.

Supporting processing includes COCO-based person/motorcycle validation, tile and crop inference for difficult large images, class-aware duplicate suppression, plate candidate refinement, and EasyOCR with multiple image enhancements. The Gradio app presents the annotated image, violation status, rider summary, and plate text.

Evaluation metrics documented in the repository are project-reported. Example images illustrate individual runs and are not an independent aggregate evaluation.
