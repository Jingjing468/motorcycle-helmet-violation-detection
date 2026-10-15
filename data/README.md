# Data

The complete 8,481-image training dataset is not stored in this repository. Project-reported split counts and class mapping are documented in [`../docs/dataset.md`](../docs/dataset.md).

## Contents

- `samples/helmet/` contains an already-used helmeted single-rider example; `samples/mixed/` contains an already-used crowded scene. They are demo inputs, not the training or evaluation dataset.
- No tracked `data.yaml` is included. The available ignored local YAML belongs to a different Roboflow Version 3 export and uses export-relative paths; it does not safely describe the reported 8,481-image experiment.

The sample images originate from the Roboflow dataset export, which identifies its license as CC BY 4.0. Source: [Helmet and Number Plate Detection for Motorbike Safety](https://universe.roboflow.com/helmet-and-number-plate-detection-project/helmet-and-number-plate-detection-for-motorbike-safety-iityz). Preserve attribution and check license terms before redistribution.
