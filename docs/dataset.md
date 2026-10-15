# Dataset

## Reported project dataset

The project experiment is reported as using **8,481 images** with the following split:

| Split | Images |
| --- | ---: |
| Train | 5,936 |
| Validation | 1,690 |
| Test | 855 |
| **Total** | **8,481** |

The dataset is in YOLO object-detection format and has four classes: `bike`, `helmet`, `no-helmet`, and `number-plate`. Class IDs are 0 through 3 in that order. Helmet and no-helmet are the primary research classes; bike and number-plate support the broader application.

The full image/label dataset is intentionally not stored in this repository. Small demonstration input images are provided in [`data/samples/`](../data/samples/) and are not a replacement for the experiment dataset.

## Local export note

The ignored local `dataset/` directory may contain a Roboflow Version 3 export. Its metadata differs from the reported 8,481-image experiment (the local copy previously documented 17,742 train, 1,690 validation, and 855 test images). Do not use that local export to claim or reproduce the reported split without first confirming which dataset version was used for the experiment.

The available Version 3 `data.yaml` points to directories relative to that export and is not a portable, safe project-root configuration. For that reason it is not copied into tracked `data/`. When supplying your own dataset, pass its valid YAML path to the evaluation or training command.

The source export identifies a CC BY 4.0 license. Preserve attribution and check the source terms before redistributing dataset content. See the source project: [Roboflow Universe — Helmet and Number Plate Detection for Motorbike Safety](https://universe.roboflow.com/helmet-and-number-plate-detection-project/helmet-and-number-plate-detection-for-motorbike-safety-iityz).
