# Model checkpoints

Checkpoint files remain in the existing `model/` tree. They are not copied into this directory, avoiding duplicate multi-megabyte weight files.

| Role | Path | Notes |
| --- | --- | --- |
| Fine-tuned custom YOLOv8 model | `model/helmet_best.pt` | Loaded by the application. Classes: bike, helmet, no-helmet, number-plate. |
| Baseline/earlier custom model | `model/archive/helmet_best_old.pt` | Retained archived checkpoint for the reported baseline comparison. |
| COCO support model | `model/coco/yolov8n.pt` | Used for supporting person/motorcycle detections. |

The paths are also defined in `config/paths.py`. Do not move or rename these weights without updating that configuration and validating the application. Reported baseline/fine-tuned metrics are in [`../docs/evaluation.md`](../docs/evaluation.md); the weights and full evaluation dataset are not copied or recalculated here.
