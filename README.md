# Parkinson's Disease Detection from MRI using Advanced Computer Vision

This project implements a novel computer vision approach to detect early signs of Parkinson's disease from MRI scans, with a focus on achieving state-of-the-art accuracy through advanced deep learning techniques.

## Quick Start Guide

1. **Setup the project**: Run `parkinson.bat setup` to install dependencies and generate synthetic data
2. **Train the model**: Run `parkinson.bat train` to train the model with optimized parameters
3. **Make predictions**: Run `parkinson.bat predict [path_to_mri_scan]` to analyze a new MRI scan
4. **Stop training**: If needed, run `parkinson.bat stop` to terminate any training in progress

Example prediction:
```
parkinson.bat predict D:\data\new_patient_scan.nii.gz
```

## Project Overview

Parkinson's disease (PD) is a neurodegenerative disorder that affects millions of people worldwide. Current diagnosis typically occurs after significant neurodegeneration has already occurred. This project aims to develop a computer vision system that can detect subtle brain changes associated with PD years before clinical symptoms appear.

### Key Features

- Multi-parametric MRI analysis combining structural and susceptibility features
- Novel transformer-based architecture specialized for neuroanatomical analysis
- Region-specific attention mechanisms focusing on key brain structures
- Contrastive learning approach for improved feature extraction
- Temporal modeling of disease progression

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Dependencies listed in `requirements.txt`

### Installation

1. Clone this repository
```
git clone [repository-url]
cd parkinsons-mri-detection
```

2. Install required packages
```
pip install -r requirements.txt
```

3. Download and prepare datasets (script provided)
```
python scripts/download_datasets.py
```

## Project Structure

```
parkinsons-mri-detection/
├── data/                   # Data storage and processing
│   ├── raw/                # Raw MRI data
│   ├── processed/          # Preprocessed data
│   └── metadata/           # Patient information and labels
├── models/                 # Model implementations
│   ├── baseline/           # Simple baseline models
│   ├── transformers/       # Transformer-based architectures
│   └── pretrained/         # Saved model weights
├── notebooks/              # Jupyter notebooks for exploration and visualization
├── scripts/                # Utility scripts
│   ├── preprocessing/      # MRI preprocessing tools
│   ├── augmentation/       # Data augmentation tools
│   └── evaluation/         # Metrics and evaluation
├── training/               # Training configurations and logs
└── visualization/          # Visualization tools and results
```

## Dataset

This project utilizes public MRI datasets including:

- **PPMI (Parkinson's Progression Markers Initiative)**: A comprehensive PD dataset with longitudinal MRI scans
- **UK Biobank**: Large population neuroimaging dataset with some PD cases
- **IXI Dataset**: Healthy control brain images

The `scripts/download_datasets.py` script handles the download and initial organization of these datasets.

## Methodology

Our approach follows these key steps:

1. **Preprocessing**: Standardization of MRI scans including registration, skull stripping, and intensity normalization
2. **Feature Extraction**: Multi-parametric analysis focusing on substantia nigra, basal ganglia, and connected pathways
3. **Model Architecture**: A specialized transformer architecture with region-specific attention mechanisms
4. **Training Strategy**: Contrastive learning approach using paired scans from the same subjects over time
5. **Evaluation**: Comprehensive evaluation using classification metrics and comparison with radiologists

## Results

Performance metrics and visualizations will be added as the project progresses.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- PPMI for providing the primary dataset
- The neuroimaging community for open-source tools
- Research groups advancing the understanding of Parkinson's disease

## References

[List of key papers and resources will be added] 