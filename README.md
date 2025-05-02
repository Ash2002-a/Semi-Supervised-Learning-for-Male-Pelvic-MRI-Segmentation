# 🧠 Semi-Supervised Deep Learning for Pelvic MRI Segmentation

This repository contains a semi-supervised deep learning framework for segmenting anatomical structures in male pelvic MRI scans. It demonstrates how to effectively use unlabeled data alongside limited labeled examples to improve medical image segmentation performance.

---

## 📚 Dataset

Uses the **Cross-Institution Male Pelvic Structures** dataset, containing:
- **589** T2-weighted MR images
- Annotations for **8 anatomical structures**

This dataset reflects a real-world clinical scenario where expert annotations are limited and expensive to acquire.

---

## 🚀 Features

- **Data Processing**: Preprocessing, augmentation, and loading scripts for MRI data
- **Supervised Baseline**: Fully supervised deep learning segmentation model
- **Semi-Supervised Learning**:
  - Pseudo-labeling
  - Consistency regularisation
  - Mean teacher approach
  - Virtual adversarial training
- **Evaluation Pipeline**: Dice coefficient, Hausdorff distance, and volume similarity metrics
- **Experiment Framework**: YAML-based config system to run and track multiple experiments

---

## 🧪 Methods Implemented

- **Pseudo-labeling**  
- **Consistency Regularization**  
- **Mean Teacher Model**  
- **Virtual Adversarial Training (VAT)**

---

## ⚙️ Installation

```bash
# Option 1: TensorFlow environment
conda create -n ssl-segmentation-tf -c conda-forge tensorflow=2.16 pillow=11.0 nibabel=5.3

# Option 2: PyTorch environment
conda create -n ssl-segmentation-pt -c conda-forge pytorch=2.4 torchvision=0.14 nibabel=5.3

# Activate your environment
conda activate ssl-segmentation-tf  # or ssl-segmentation-pt
```
## ▶️ Usage
Place the dataset into the data/ directory

Run experiments using config files:

```bash
python main.py --config configs/experiment1.yaml
```

## 📊 Results
- Semi-supervised models significantly outperformed fully-supervised baselines

- Clear correlation between labeled/unlabeled data proportions and performance

- Visualisations of segmentation results across different methods

## 📦 Requirements
Python 3.8+

TensorFlow 2.16+ or PyTorch 2.4+

nibabel 5.3+

pillow 11.0+

## 🔮 Future Work
- Explore additional semi-supervised learning strategies

- Extend segmentation to other anatomical regions

- **Integrate active learning techniques for efficient data labeling

