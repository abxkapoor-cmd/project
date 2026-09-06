# Vehicle Detection in Aerial Imagery (Mask R-CNN)

A fine-tuned Mask R-CNN model that detects vehicles in aerial imagery, trained on the COWC (Cars Overhead With Context) dataset. This project replicates the core computer vision technique behind real satellite-based alternative-data strategies used in finance — companies like RS Metrics have sold retail parking-lot car counts to hedge funds as a proxy for footfall and earnings signals since 2011.

## Overview

- **Task:** Object detection (vehicle localization) in top-down aerial imagery
- **Architecture:** Mask R-CNN (ResNet-50 FPN backbone), pretrained on COCO, fine-tuned on COWC
- **Framework:** PyTorch / torchvision
- **Dataset:** COWC — Toronto region patches
- **Hardware:** NVIDIA RTX 3050 (4GB), CUDA-accelerated training

## Results

| Metric | 3 epochs | 10 epochs |
|---|---|---|
| mAP (IoU 0.5–0.95) | 0.43 | 0.59 |
| mAP@50 | — | 0.60 |
| mAP@75 | — | 0.60 |
| mAR@100 (recall) | 0.83 | 0.76 |

Evaluated on a held-out, non-overlapping 500-image test split (seeded, no overlap with the 2,000-image training subset).

## Method

### Data challenge

COWC's publicly available "detection" patches are actually classification-labeled, not box-annotated — each 256×256 patch is labeled `car.` or `neg.` based on whether a vehicle is centered in it, with no ground-truth bounding box coordinates provided. Since Mask R-CNN requires box/mask supervision, this dataset couldn't be used directly out of the box.

### Solution: size-prior bounding boxes

Because COWC's positive patches are deliberately cropped so the labeled vehicle sits at the center, an approximate bounding box could be reconstructed without manual annotation:

- **Position:** fixed at the patch center (justified by COWC's known construction method)
- **Size:** fixed at 40×15 px, based on COWC's documented typical vehicle dimensions

**Limitation (stated honestly):** this produces one fixed box size for every vehicle, rather than each car's true dimensions. This caps achievable precision compared to a dataset with real hand-drawn boxes (e.g. VEDAI), and is the most likely lever for improving results further.

### Training

- Custom PyTorch `Dataset` class parsing COWC filenames directly (`car.` / `neg.` prefix → label)
- 2,000-image random training subset (of 36,544 available), 500-image held-out evaluation subset, both seeded for reproducibility
- SGD optimizer (lr=0.005, momentum=0.9, weight_decay=0.0005), batch size 2
- Per-epoch checkpointing (`checkpoint_epoch_N.pth`)
- Evaluation via torchmetrics' `MeanAveragePrecision`

## Repository structure

```
cowc_dataset.py          # Custom Dataset class: loads COWC patches, derives boxes/masks
train.py                 # Training loop, checkpointing
eval.py                  # Loads a checkpoint, runs evaluation on held-out data
checkpoint_epoch_*.pth   # Saved model weights per epoch
```

## How to run

```bash
pip install torch torchvision torchmetrics

python train.py   # trains and saves checkpoints
python eval.py     # evaluates a chosen checkpoint on held-out data
```

## Limitations & future work

- **Approximate boxes:** fixed-size boxes rather than true per-vehicle dimensions (see above)
- **Training scale:** trained on ~5.5% of available COWC data (2,000 / 36,544 images), 10 epochs — both are levers for further improvement given more compute time
- **Domain gap:** the model is trained on COWC's ~15cm/pixel resolution; applying it to lower-resolution public imagery (e.g. NAIP, ~0.6–1m/pixel) would require resolution correction, since a vehicle occupies a fundamentally different pixel footprint at each scale
- **Financial validation:** a genuine alt-data signal (as used commercially) requires aggregating counts across hundreds of store locations against company-wide revenue — a single-location count cannot be meaningfully validated against total company revenue, so this project scopes itself to the detection methodology rather than claiming a financial signal

## Acknowledgements

- [COWC Dataset](https://gdo152.llnl.gov/cowc/) — Cars Overhead With Context
- torchvision Mask R-CNN implementation
