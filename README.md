# ROSE-OCTA-segmentation

Deep learning framework for retinal OCTA vessel segmentation and quantitative biomarker analysis on the ROSE dataset.

## Overview

This repository contains the code developed for retinal vessel segmentation from Optical Coherence Tomography Angiography (OCTA) images.

The project investigates deep learning-based segmentation methods for accurate extraction of retinal vascular structures and subsequent quantitative analysis through morphological biomarkers.

The framework includes:

- OCTA image preprocessing
- deep learning-based vessel segmentation
- model training and evaluation
- topology-aware segmentation assessment
- retinal vascular biomarker extraction

## Pipeline

The proposed pipeline consists of:

1. Dataset preparation and preprocessing
2. Deep learning model training
3. Segmentation inference
4. Quantitative evaluation using segmentation metrics
5. Extraction of vascular biomarkers from predicted vessel maps

## Repository Structure

```
ROSE-OCTA-segmentation/
│
├── src/
│   ├── datasets/          # Dataset loading and preprocessing
│   ├── models/            # Neural network architectures
│   ├── training/          # Training and validation pipelines
│   ├── evaluation/        # Metrics and biomarker computation
│   ├── postprocessing/    # Segmentation refinement methods
│   └── utils/             # Utility functions
│
├── notebooks/             # Jupyter notebooks
├── configs/               # Configuration files
├── figures/               # Generated figures
├── results/               # Experimental results
│
├── requirements.txt       # Python dependencies
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/USERNAME/ROSE-OCTA-segmentation.git
cd ROSE-OCTA-segmentation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Dataset

The datasets are not included in this repository.

The experiments are performed on publicly available retinal OCTA datasets. Dataset paths should be configured locally before running the training and evaluation scripts.

## Training

Training scripts will be provided to reproduce the experiments described in the paper.


## Evaluation

Segmentation performance is evaluated using:

- Dice Similarity Coefficient (DSC)
- clDice
- sensitivity and specificity
- topology-aware metrics

The framework also includes quantitative analysis of retinal vascular biomarkers extracted from segmentation maps.

## Results

Experimental results and generated figures will be added in the `results/` and `figures/` folders.

## Citation

If you use this code, please cite:

```
Paper citation will be added after publication.
```

## License

This project is released for academic research purposes.