# Evaluation

The figures below are the project-reported baseline/fine-tuned comparison supplied for the project. They are transcribed for documentation; this repository restructuring did not rerun evaluation or regenerate the metrics.

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

The fine-tuned model has higher reported recall and mAP values, including the no-helmet metrics, while precision is slightly lower. These aggregate metrics summarize the supplied evaluation; they do not imply perfect detection in individual images. The no-helmet class is central to the project objective.

## Metric definitions

- **Precision:** proportion of predicted positives that are correct.
- **Recall:** proportion of ground-truth positives that are found.
- **mAP@50:** mean average precision at IoU 0.50.
- **mAP@50:95:** mean average precision averaged over IoU thresholds 0.50 to 0.95.

## Re-evaluation

The CLI accepts a compatible external YOLO dataset YAML:

```bash
python scripts/evaluate_model.py --data path/to/data.yaml
```

Supply the exact dataset version used for comparison to make results comparable. The full dataset and the original raw metric exports are not included, so this repository cannot independently reproduce the table from the current tracked files alone.
