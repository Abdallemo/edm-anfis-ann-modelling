# EDM ANFIS-ANN Modelling

A comparative framework evaluating Adaptive Neuro-Fuzzy Inference Systems (ANFIS) and Artificial Neural Networks (ANN) for predicting surface roughness in Electrical Discharge Machining (EDM).

This project investigates the limits of machine learning on small-sample experimental datasets (e.g., Taguchi L18/L27 arrays), proving the necessity of Hybrid Learning (LSE + Gradient Descent) and Ridge Regularization over pure deep-learning approaches for noisy, physical manufacturing data.

## Requirements

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (for strict dependency locking and environment management)

## Installation

Clone the repository and install the locked dependencies via `uv`:

```bash
git clone [https://github.com/Abdallemo/edm-anfis-ann-modelling.git](https://github.com/Abdallemo/edm-anfis-ann-modelling.git)
cd edm-anfis-ann-modelling
uv sync

```

## Architecture

The project consists of three core computational systems:

1. **Hybrid ANFIS (`anfis.py`)**: A custom PyTorch implementation of a Sugeno ANFIS. It bypasses standard L-BFGS convergence failures on small datasets by utilizing a MATLAB-style Hybrid Learning Algorithm. Premise parameters (Bell curves) are tuned via Gradient Descent, while consequent parameters (linear equations) are solved instantly via regularized Least Squares Estimation (LSE).
2. **Deep Learning ANN (`ann.py`)**: A baseline `MLPRegressor` configured with automated polynomial feature expansion to map physical machine interactions (e.g., Spark Energy) prior to training.
3. **LOOCV Pipeline (`experiment.py`)**: A strict Leave-One-Out Cross-Validation engine designed to prevent data leakage in 30-row datasets. It automatically executes massive hyperparameter grid searches and exports multi-sheet performance matrices (Best vs. Failure states) to Excel.

## Usage

### Interactive Prediction (GUI)

To launch the PySide6 graphical interface for model building/interaction:

```bash
uv run predict-gui.py

```

## Dataset Configuration

The pipeline is statically typed to expect the following machine parameters as inputs:

- **Dataset 1 (Chaotic/Noisy):** `volt`, `ip`, `ton`, `toff`
- **Dataset 2 (Taguchi L27):** `ton`, `duty_cycle`, `peak_current`, `voltage`

Models will automatically apply scaling and polynomial expansions based on their serialized state during inference.

## Acknowledgements

Developed under the academic supervision of Dr. Nurezayana Zainal and Dr. Nur Ariffin ZIN to facilitate ongoing research in EDM optimization.

```
