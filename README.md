# OCTA-Segmentation-ROSE

Deep learning framework for retinal vessel segmentation in Optical Coherence Tomography Angiography (OCTA) images, with topology-aware evaluation and vascular biomarker analysis.

## Overview

This repository contains the implementation of a deep learning pipeline for retinal vessel segmentation from OCTA images.

The framework focuses on accurate extraction of retinal vascular structures while preserving vascular topology, enabling quantitative analysis through segmentation metrics and morphological biomarkers.

The pipeline includes:

- OCTA image preprocessing
- Attention U-Net based vessel segmentation
- multi-stage training strategy
- topology-aware evaluation
- morphological post-processing analysis
- vascular biomarker extraction

## Methodology

The proposed training strategy follows a progressive domain adaptation pipeline:

1. **Generic vessel segmentation pre-training**

   The model is initially trained on the FIVES fundus vessel segmentation dataset to learn general retinal vessel representations.

2. **OCTA domain adaptation**

   The model is adapted to OCTA characteristics using the OCTA-500 dataset with fused large vessel and capillary annotations.

3. **ROSE-1 fine-tuning**

   The adapted model is fine-tuned on the ROSE-1 OCTA dataset for superficial and full retinal vascular segmentation.

## Model Architecture

The implemented segmentation model is:

- Attention U-Net architecture
- ResNet34 encoder backbone
- ImageNet pretrained encoder weights
- SCSE decoder attention mechanism
- Binary vessel segmentation output

Training uses a hybrid loss function:

- 70% Dice Loss
- 30% Binary Cross Entropy Loss

## Evaluation

Segmentation performance is evaluated using:

- Dice Similarity Coefficient (DSC)
- clDice topology-aware metric
- category-specific Dice analysis
- morphological post-processing comparison

Topology-aware analysis includes:

- skeleton extraction
- vessel connectivity evaluation
- morphological filtering
- post-processing grid search

## Repository Structure

```
OCTA-Segmentation-ROSE/

├── src/
│   ├── datasets/              # Dataset loading and preprocessing
│   ├── models/                # Neural network architectures
│   ├── training/              # Training and validation routines
│   ├── evaluation/            # Metrics and topology analysis
│   ├── losses/                # Loss functions
│   └── utils/                 # Utility functions
│
├── scripts/
│   ├── train_fives.py         # FIVES pre-training
│   ├── train_octa500.py       # OCTA-500 domain adaptation
│   ├── train_rose.py          # ROSE-1 fine-tuning
│   ├── evaluate_rose.py       # Quantitative evaluation
│   ├── visualize_results.py   # Result visualization
│   └── test_setup.py          # Repository setup test
│
├── configs/
│   └── paths.yaml             # Local dataset configuration
│
├── checkpoints/               # Saved model weights
├── results/                   # Evaluation outputs
├── figures/                   # Generated figures
│
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/lipantar8/OCTA-Segmentation-ROSE.git
cd OCTA-Segmentation-ROSE
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset Configuration

Datasets are not included in this repository.

The required datasets are:

- FIVES
- OCTA-500
- ROSE-1

Dataset locations must be specified locally in:

```
configs/paths.yaml
```

Example:

```yaml
datasets:

  ROSE:
    SVC:
      images: /path/to/images
      masks: /path/to/masks
```

## Training

The complete training pipeline consists of three stages.

### 1. FIVES pre-training

```bash
python scripts/train_fives.py
```

Output:

```
checkpoints/best_model_FIVES.pth
```

### 2. OCTA-500 domain adaptation

```bash
python scripts/train_octa500.py
```

Output:

```
checkpoints/best_model_OCTA500.pth
```

### 3. ROSE-1 fine-tuning

```bash
python scripts/train_rose.py
```

Output:

```
checkpoints/best_model_ROSE_fold*.pth
```

## Evaluation

Run:

```bash
python scripts/evaluate_rose.py
```

Generated results:

```
results/

├── evaluation_results.csv
└── topology_results.csv
```

Visualization:

```bash
python scripts/visualize_results.py
```

Figures are saved in:

```
results/figures/
```

## Reproducibility

The repository is designed to reproduce the complete experimental pipeline.

Due to dataset licensing restrictions, images and annotations are not distributed with this repository.

Users must download the datasets independently and configure local paths before running experiments.

## Citation

If you use this repository, please cite:

```
Citation will be added after publication.
```

## License

This project is released for academic research purposes.