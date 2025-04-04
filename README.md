# Restricted Boltzmann Machine (RBM)

This repository contains a custom implementation of a Restricted Boltzmann Machine (RBM) using NumPy. It is built from scratch for better understanding and customizability, trained on Metropolis Monte Carlo data from artificial spin ice (ASI) systems, specifically square and pinwheel geometries. The RBM aims to capture statistical correlations and identify structural defects in ASI systems under various conditions.

## Features
- Supports training with momentum-based gradient descent.
- Conditional momentum adjustment based on epoch count for enhanced learning performance.
- Configurable learning rate, batch size, and training epochs.
- Command-line interface (CLI) with `typer` for easy configuration.
- Modular and well-documented codebase for ease of maintenance.

## Installation

```bash
pip install typer numpy
```

## Usage

The code provides a simple command-line interface (CLI) via `typer`. To run the training process, use:

```bash
python your_script.py --file-path '/path/to/your/dataset.npy' --epochs 15000 --lr 0.0001 --batch-size 50
```

### Arguments
- `--file-path`: Path to the `.npy` file containing your training data.
- `--epochs`: Number of training epochs. (Default: 15000)
- `--lr`: Learning rate. (Default: 0.0001)
- `--batch-size`: Size of training batches. (Default: 50)

### Momentum Adjustment
- Momentum is dynamically adjusted based on the epoch:
  - **Epoch <= 1500:** High momentum (0.9) to accelerate learning.
  - **Epoch > 1500:** Lower momentum (0.3) for more refined updates.

## Example

```bash
python your_script.py --file-path '/home/mahdis/coded_6lat_T=3.5_600k.npy' --epochs 15000 --lr 0.0001 --batch-size 50
```

## Output
- Training progress is printed to the console with epoch number, error, and elapsed time.
- Model parameters and training performance can be further processed as needed.

## Dependencies
- `numpy`
- `typer`

