# Model weights

The application loads the fine-tuned custom YOLOv8 checkpoint from `helmet_best.pt` and the COCO support model from `coco/yolov8n.pt` (paths are defined in `config/paths.py`). The baseline/earlier custom checkpoint is kept at `archive/helmet_best_old.pt`. Checkpoint roles and comparison context are also summarized in [`../checkpoints/README.md`](../checkpoints/README.md).

The custom detector class mapping is:

| ID | Class | Project role |
| ---: | --- | --- |
| 0 | `bike` | Motorcycle support |
| 1 | `helmet` | Main feature |
| 2 | `no-helmet` | Main feature |
| 3 | `number-plate` | Supporting plate detection |

`helmet_best.pt` is the app's current fine-tuned/demo model. `helmet_best_old.pt` is the earlier baseline/archived model used for the reported comparison. Large checkpoints are stored only once; do not copy them under `checkpoints/`.
