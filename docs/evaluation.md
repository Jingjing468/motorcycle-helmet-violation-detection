# Evaluation

The following values are already recorded in the project README. They are reproduced as project-reported metrics, not newly measured by the restructuring work.

| Metric | Earlier checkpoint | Fine-tuned checkpoint |
| --- | ---: | ---: |
| Precision | — | 0.918 |
| Recall | — | 0.889 |
| mAP@50 | 0.942 | 0.942 |
| mAP@50:95 | 0.675 | 0.690 |

Per-class mAP@50:95 values are in the README. The saved custom weights are checkpoints of the same YOLOv8 architecture, not two different architectures. Use `python scripts/evaluate_model.py --data path/to/data.yaml` to run validation and report metrics on a supplied dataset.
