# Dataset

## Reported project dataset

The project experiment is reported as using **8,481 images** with the following split:

| Split | Images |
| --- | ---: |
| Train | 5,936 |
| Validation | 1,690 |
| Test | 855 |
| **Total** | **8,481** |

The source is Roboflow Universe, **Version 4**, in YOLO object-detection format. Its four class IDs are `0 bike`, `1 helmet`, `2 no-helmet`, and `3 number-plate`. Helmet and no-helmet are the primary research classes; bike and number-plate support the broader application.

## Image and label preparation

Project training used 640 × 640 input images. The dataset export carries object bounding boxes and class IDs in YOLO detection format; the model uses those labels for the four classes above. `data/samples/` contains a small helmeted-rider input and a mixed/crowded input for demos only. It does not include training labels or the full export.

The repository does not contain the Version 4 export YAML or a full annotation audit trail. Do not assume additional augmentation, review, or split-generation details beyond the recorded facts above.

The full image/label dataset is intentionally not stored in this repository. Small demonstration input images are provided in [`data/samples/`](../data/samples/) and are not a replacement for the experiment dataset.

## Local export note

The ignored local `dataset/` directory contains a Roboflow Version 3 export. Its metadata differs from the reported 8,481-image Version 4 experiment (17,742 train, 1,690 validation, and 855 test images). Do not use that local export to claim or reproduce the reported Version 4 split.

The available Version 3 `data.yaml` points to directories relative to that export and is not a portable, safe project-root configuration. For that reason it is not copied into tracked `data/`. When supplying your own dataset, pass its valid YAML path to the evaluation or training command.

The source export identifies a CC BY 4.0 license. Preserve attribution and check the source terms before redistributing dataset content. See the source project: [Roboflow Universe — Helmet and Number Plate Detection for Motorbike Safety](https://universe.roboflow.com/helmet-and-number-plate-detection-project/helmet-and-number-plate-detection-for-motorbike-safety-iityz).
