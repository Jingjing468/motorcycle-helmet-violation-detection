# Notebooks

These notebooks use the existing YOLOv8 tooling and require the project dataset to be available locally; the full dataset is not stored here.

- [`training.ipynb`](training.ipynb) provides baseline training and difficult-no-helmet fine-tuning cells using the recorded 640 / batch 16 / 40 epoch settings. It requires `HELMET_DATA_YAML`; set `HELMET_FINETUNE_DATA_YAML` for the fine-tuning data config.
- [`evaluation.ipynb`](evaluation.ipynb) validates the archived baseline and app fine-tuned weights, reports aggregate and per-class metrics, and displays the separate project-reported table from `results/metrics/comparison.csv`.

The previous `helmet_violation_detection.ipynb` file contained zero bytes and no usable cells. It was replaced with these executable workflows. There is no experiments notebook because raw threshold sweeps and per-example experiment logs are not available to populate one accurately.
