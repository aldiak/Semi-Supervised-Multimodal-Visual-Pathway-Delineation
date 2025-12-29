# Cross-sequence semi-supervised learning for multi-parametric MRI-based visual pathway delineation

## Overview

This repository implements a semi-supervised segmentation framework for visual pathway structures (cranial nerve II) segmentation using multi-modal MRI data (T1-weighted and Fractional Anisotropy maps). The core method is based on a **Decompose** architecture with decomposition regularization and a reliable Ensemble Mean Teacher framework for semi-supervised learning.

Key features:
- Semi-supervised training with limited labeled data
- 5-fold cross-validation
- Inference and quantitative evaluation on test sets
- Supported metrics: Dice, Jaccard (IoU), Hausdorff Distance 95 (HD95), Mean Surface Distance (MSD)

Supported datasets: HCP, MDM, MMD (configurable via paths).

## Repository Structure
```tree
.
├── ours3_all.py                  # Training script
├── evaluate.py                   # Inference + evaluation script
├── config_2d.py                  # Global configuration (patch size, paths, etc.)
├── dataloader.py                 # Dataset classes for training and testing
├── models/
│   ├── Decompose.py              # Main model with decomposition regularization
│   ├── Decompose_wdcp.py         # Variant of Decompose
│   └── SAnet.py                 # Alternative T1-FA fusion UNet (t1safaFuseUNet1)
├── seg_metrics/                  # External metric library (seg_metrics)
├── run_on_segmentation.sh        # Recommended bash script for easy training/evaluation
└── README.md                     # This file
```

## Requirements

- Python ≥ 3.8
- PyTorch (tested with 1.13+)
- torchvision
- nibabel
- numpy
- scipy
- tqdm
- tensorboardX
- joblib
- scikit-learn
- medpy (for HD95)
- seg_metrics (https://github.com/hubutui/seg_metrics)


## Data Organization
### Training Data
```tree
/path/train_data/
├── x_t1_data/      # T1 slices (.nii.gz)
├── x_fa_data/      # FA slices (.nii.gz)
└── y_data/         # Binary label slices (.nii.gz)
```

### Test Data (Slice-based for inference)
```tree
/path/test_data/
└── test_<subject_id>/
    ├── x_t1_data/
    ├── x_fa_data/
    └── y_data/
```

### Test Data (Full volumes for evaluation)
```tree
/path/test_imgs/   # or config_2d.test_imgs_path
└── <subject_id>/
    ├── <subject_id>_ON-T1.nii.gz
    ├── <subject_id>_ON-mask.nii.gz
    └── <subject_id>_ON-label.nii.gz
```

## Usage

### Training
#### provide the different arguments (data path and hyper-parameters) defined in ours3_all.py, then
python ours3_all.py    

### Evaluation:
#### Edit model_path, data paths, and pred_dir in evaluate.py, then:
python evaluate.py 

## Citation
### Please cite our work accepted in Physics in Medicine and Biology, please cite it accordingly when published.
