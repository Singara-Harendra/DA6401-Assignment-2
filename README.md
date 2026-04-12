# Multi-Task Perception on Oxford-IIIT Pet Dataset 🐾

**Report Link:** [https://api.wandb.ai/links/da25m028-indian-institute-of-technology-madras/tcvd9dn0](https://api.wandb.ai/links/da25m028-indian-institute-of-technology-madras/tcvd9dn0)

This repository contains a unified deep learning framework for solving three distinct computer vision tasks simultaneously on the Oxford-IIIT Pet dataset: **Classification** (breed recognition), **Localization** (bounding box regression), and **Segmentation** (pixel-level trimap prediction).

## 🚀 Overview

The project implements a shared-backbone architecture using a VGG11 encoder. By sharing features, the model learns more robust representations across tasks while reducing overall computational overhead.

- **Task 1: Classification** - Identifying one of 37 pet breeds.
- **Task 2: Localization** - Predicting the bounding box `(x, y, w, h)` of the pet's head.
- **Task 3: Segmentation** - Classifying each pixel into foreground, background, or boundary.
- **Task 4: Unified Model** - A single multi-head model that performs all three tasks in one forward pass.

## 🏗️ Project Structure

```text
├── data/
│   └── pets_dataset.py     # Custom Dataset class with train-val split
├── models/
│   ├── __init__.py
│   ├── layers.py           # Custom Inverted Dropout implementation
│   ├── vgg11.py            # VGG11 Encoder (backbone) with BatchNorm
│   ├── classification.py   # Classification head and full model
│   ├── localization.py     # Localization head and full model
│   ├── segmentation.py     # U-Net style decoder for segmentation
│   └── multitask.py        # Unified shared-backbone model
├── losses/
│   ├── __init__.py
│   └── iou_loss.py         # Custom IoU Loss for bounding box regression
├── checkpoints/
│   └── checkpoints.md      # Directory for storing .pth model weights
├── train.py                # Main training script with W&B support
├── inference.py            # Script for running predictions on single images
├── README.md               # Project documentation
└── requirements.txt        # Project dependencies

## 🛠️ Installation

Ensure you have Python 3.8+ installed. Install the dependencies using:

```bash
pip install -r requirements.txt
```

## 🏋️ Training

You can train individual tasks or the unified model using `train.py`. The script supports automatic resuming from checkpoints and integrates with Weights & Biases for logging.

**To train the Multi-Task model:**
```bash
python train.py --task multitask --epochs 20 --batch_size 32 --lr 1e-3
```

**Key Training Arguments:**
- `--task`: `classification`, `localization`, `segmentation`, or `multitask`.
- `--data_root`: Path to the extracted Oxford-IIIT Pet dataset.
- `--dropout_p`: Probability for the custom dropout layers (default: 0.5).

## 🔍 Inference

Run inference on a single image using a trained checkpoint:

```bash
python inference.py --task multitask --image path/to/pet.jpg --ckpt checkpoints/multitask/best.pth
```

## 📊 Evaluation Metrics

The framework evaluates performance using standard metrics consistent with competition benchmarks:
- **Classification:** Macro-F1 Score.
- **Localization:** Accuracy at IoU >= 0.5.
- **Segmentation:** Macro-Dice Coefficient.
```
