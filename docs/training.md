# Training and Fine-Tuning

Project notes report that YOLOv8 training and fine-tuning used Google Colab with an NVIDIA Tesla T4. Fine-tuning continued the existing checkpoint for 10 additional epochs and added difficult examples, especially rear views, crowded scenes, small heads, and multiple riders.

The repository has no populated training notebook or complete training configuration. Batch size, optimizer settings, exact splits, random seeds, and the full augmentation configuration are therefore not asserted here. The current application loads `model/helmet_best.pt`; the earlier checkpoint is retained in `model/archive/`.
