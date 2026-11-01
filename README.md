# CO2 impact on convolutional network model training for autonomous driving through behavioral cloning

This repository contains the implementation and experimental workflow associated with the paper:

"CO2 impact on convolutional network model training for autonomous driving through behavioral cloning"

published in the Advanced Engineering Informatics Journal.

## Overview

This project studies how deep learning models can learn end-to-end autonomous driving behavior from camera data using behavioral cloning. The system predicts steering commands from visual input and is trained on simulator-generated driving data.

The work focuses on:

- behavioral cloning for steering prediction,
- comparison of CNN architectures,
- preprocessing and augmentation strategies,
- training efficiency and computational cost,
- deployment in a driving simulator for validation.

## Research objective

The main goal is to design and evaluate a vision-based autonomous driving system that learns steering behavior directly from demonstrations. The repository supports experiments comparing different convolutional neural network architectures and studying their efficiency in terms of training cost and performance.

## Repository structure

```text
repo_name/
├── README.md
├── requirements_TFM.txt
├── requirements_TFM_Conda.txt
├── Scripts_TFM/
│   ├── __init__.py
│   ├── main.py
│   ├── models.py
│   ├── preprocessing.py
│   ├── data_utils.py
│   ├── visualization.py
│   ├── TFM_uoc_adbc_PilotNet_VGG.py
│   ├── drive_v1.py
│   ├── drive_v2.py
│   ├── drive4.py
│   ├── model_nor.h5
│   ├── model_VGG.h5
│   └── README.md
└── trained_models/            # generated during training
```

## Main components

- `Scripts_TFM/main.py`: training entry point for the project.
- `Scripts_TFM/models.py`: CNN model definitions, including PilotNet-inspired and VGG-style architectures.
- `Scripts_TFM/preprocessing.py`: image preprocessing and data augmentation utilities.
- `Scripts_TFM/data_utils.py`: dataset preparation, balancing, splitting, and batch generation.
- `Scripts_TFM/visualization.py`: visualization utilities for steering distributions and sample images.
- `Scripts_TFM/drive_v1.py`, `drive_v2.py`, `drive4.py`: simulator inference scripts.
- `Scripts_TFM/TFM_uoc_adbc_PilotNet_VGG.py`: legacy exploratory/training script retained for reference.

## Methodology

### Data

The project uses driving logs generated from a simulator and typically includes:

- center, left, and right camera images,
- steering angle,
- throttle,
- brake,
- speed.

### Preprocessing and augmentation

The pipeline includes:

- image cropping,
- conversion to YUV color space,
- Gaussian blur,
- resizing to CNN input dimensions,
- pixel normalization,
- brightness adjustment,
- horizontal flipping,
- translation and shadow augmentation.

### Model architectures

Two main CNN families are used:

- PilotNet-inspired network,
- VGG-style architecture.

Both models predict a continuous steering value from a single image.

## Quick start

### 1. Install dependencies

Using pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements_TFM.txt
```

Using conda:

```bash
conda create --name tfm_uoc python=3.8
conda activate tfm_uoc
pip install -r requirements_TFM.txt
```

### 2. Train the model

```bash
python Scripts_TFM/main.py --data-dir /path/to/dataset --model vgg --epochs 20 --batch-size 16 --output-dir ./trained_models
```

Available model choices:

- `vgg`
- `nvidia`

### 3. Run from Python

```python
from Scripts_TFM.main import main

if __name__ == "__main__":
    main()
```

## Simulation and inference

The simulator scripts load a trained model and use live telemetry to predict steering commands in real time.

```bash
cd Scripts_TFM
python drive_v1.py
```

```bash
cd Scripts_TFM
python drive_v2.py
```

```bash
cd Scripts_TFM
python drive4.py
```

## Outputs

Training produces:

- trained model checkpoints in `.h5` format,
- TensorBoard logs under `seq_logs`,
- saved artifacts under `trained_models/`.

## Citation

If you use this work in your research, please cite the original paper:

CO2 impact on convolutional network model training for autonomous driving through behavioral cloning. Advanced Engineering Informatics Journal.

## License

This repository is provided for research and reproducibility purposes. Please cite the original publication when reusing code, results, or methodology.

## Contact

For collaboration or questions about the project, please reach out through the corresponding publication or repository maintainer.
