# Gradio Demo Guide

## Start the app

From the repository root, activate the project environment and run:

```bash
python app/app.py
```

Open the local Gradio URL printed in the terminal. The app loads its model files from the paths in `config/paths.py`. For installation instructions, see the root README.

## Suggested presentation sequence

1. Upload an image with a helmeted rider. Review the annotated image, detected-object crop, rider status, and confidence.
2. Upload an image with a rider not wearing a helmet. Review the validated no-helmet result and violation status.
3. Upload a mixed/crowded scene and review multiple rider statuses and confidence values.
4. Review supporting motorcycle detections, number-plate detections, and any EasyOCR output.

Use examples from `results/figures/examples/`, `data/samples/helmet/`, and `data/samples/mixed/`. Results can vary by image. In particular, an absent helmet detection does not itself establish a violation; the app requires a validated no-helmet detection associated with a motorcycle occupant.

## Reading the outputs

- The annotated result shows accepted public detections and confidence labels.
- The detected-object gallery shows crops from those same accepted detections.
- The summary lists motorcycle, helmet, no-helmet, and plate counts, followed by rider status.
- Plate text is produced by EasyOCR and may be unreadable or incorrect. Treat it as an experimental aid.
- **Clear** removes the current image and outputs.

The application currently handles still images. Video or live-camera use is future work.
