"""
Module ann provides a high-level framework for modeling Electrical Discharge
Machining (EDM) surface roughness (Ra) using Artificial Neural Networks.

Design Philosophy:
The module follows an encapsulation pattern where preprocessing (scaling) is
bound directly to the model instance. This prevents 'data leakage' and
ensures that production inference exactly matches the training environment.

Data Context:
- Dataset 1: Characterized by raw experimental noise; targets stability (~0.67 R2).
- Dataset 2: Highly structured Taguchi L27 design; targets high precision (~0.99 R2).
"""

import itertools
import os
from typing import Literal, Self, TypedDict, Union

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class ModelState(TypedDict):
    model: MLPRegressor
    scaler: StandardScaler
    features: list[str]


class NeuralNetwork:
    """
    NeuralNetwork provides an encapsulated wrapper around sklearn's MLPRegressor.

    Operational Requirements:
    1. Features must be passed as pandas DataFrames to maintain consistency in
       feature names and prevent StandardScaler warnings.
    2. Model persistence is handled via a state dictionary containing both
       the fitted scaler and the model weights.

    Implementation Note:
    Unlike raw MLPRegressor, this class manages its own internal state for
    StandardScaler. This ensures that when a model is loaded for inference,
    it applies the exact same mean/variance scaling used during the
    original training session.
    """

    def __init__(
        self,
        hidden_layers: tuple[int, ...] = (5,),
        activation="relu",
        max_iter=10000,
        alpha=0.0001,
    ) -> None:
        """
        Initializes the model with specific architecture and regularization.

        Args:
            hidden_layers: Topology of the network (e.g., (8, 4) for two layers).
            activation: Activation function ('logistic', 'tanh', 'relu').
            alpha: L2 penalty (regularization term) to prevent overfitting in
                   noisy experimental data.
        """
        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layers,
            activation=activation,
            solver="lbfgs",
            alpha=alpha,
            max_iter=max_iter,
            random_state=42,
        )

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """
        Trains the internal model using the provided machining parameters.

        Note:
        Automatically executes fit_transform on the internal scaler. If data
        contains NaNs or non-numeric types, a ValueError is raised.
        """
        #
        try:
            self.feature_names = X_train.columns.tolist()
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_train_scaled, y_train)
        except ValueError as e:
            raise ValueError(
                f"Training failed due to invalid data format: {e}"
            ) from None

    def predict(self, x_test: pd.DataFrame):
        """
        Predicts surface roughness for a given set of machining inputs.

        Operational Requirement:
        The input DataFrame must contain the same columns (in the same order)
        as the training set.

        Returns:
            The predicted Ra value as a float.
        """
        try:
            x_test_scaled = self.scaler.transform(x_test)
            predictions = self.model.predict(x_test_scaled)

            return float(predictions[0])
        except NotFittedError:
            raise NotFittedError(
                "The model must be trained or loaded before making predictions."
            ) from None
        except ValueError as e:
            raise ValueError(
                f"Prediction failed due to mismatched input features: {e}"
            ) from None

    def save(self, filepath: str) -> None:
        """Saves both the model and the scaler into a single file."""
        try:
            state: ModelState = {
                "model": self.model,
                "scaler": self.scaler,
                "features": getattr(self, "feature_names", []),
            }
            joblib.dump(state, filepath)

            print(f"Network successfully saved to {filepath}")
        except OSError as e:
            raise OSError(
                f"Could not save the model to '{filepath}'. Check directory permissions: {e}"
            ) from None

    @classmethod
    def load(cls, filepath: str) -> Self:
        """
        Reconstructs a NeuralNetwork instance from a binary state file.

        Deployment Note:
        Requires a valid .pkl file generated by the .save() method. The file
        must contain the 'model' and 'scaler' keys or a KeyError will be raised.
        """
        try:
            state: ModelState = joblib.load(filepath)

            if "model" not in state or "scaler" not in state or "features" not in state:
                raise KeyError(
                    "The loaded file is missing the 'model' or 'scaler' state."
                )

            network = cls()

            network.model = state["model"]
            network.scaler = state["scaler"]
            network.feature_names = state.get("features", [])

            return network

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Error: The model file '{filepath}' does not exist."
            ) from None
        except KeyError as e:
            raise KeyError(
                f"Error: The file is corrupted or not a valid NeuralNetwork state: {e}"
            ) from None
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while loading the model: {e}"
            ) from None


def evaluate(prediction: list[float], y_true):
    mse = mean_squared_error(y_true, prediction)
    rmse: np.float64 = np.sqrt(mse)
    mae = mean_absolute_error(y_true, prediction)
    r2 = r2_score(y_true, prediction)
    return rmse, mae, r2


class ParamGrid(TypedDict):
    hidden_layers: list[tuple[int, ...]]
    activation: list[Literal["relu", "logistic", "tanh"]]
    alpha: list[float]


class ExperimentRunner:
    """
    ExperimentRunner is the orchestration layer for model validation and
    hyperparameter optimization.

    Operational Requirements:
    Requires the 'evaluate' helper function to compute RMSE, MAE, and R2 metrics.
    Designed to compare two primary validation strategies:
    1. LOOCV: Leave-One-Out Cross-Validation. Recommended for small datasets
       (<50 rows) where every data point is critical for evaluation.
    2. K-Fold: Standard cross-validation. Useful for verifying model stability
       across different data shuffles.

    Storage Note:
    Automatically tracks 'best_predictions' and 'best_actuals' during a
    grid_search session to allow for one-click results generation (Plots and CSVs).
    """

    def __init__(self, X, y):
        """
        Args:
            X: Input machining parameters (Volt, Ip, Ton, etc.).
            y: Target surface roughness values (Ra).
        """
        self.X = X
        self.y = y
        self.loo = LeaveOneOut()
        self.best_predictions = []
        self.best_actuals = []

    def grid_search(
        self,
        param_grid: ParamGrid,
        split: Literal["loocv", "kfold"] = "loocv",
    ):
        """
        Executes an exhaustive search over the parameter space to find the
        global optimum for a specific dataset.

        Validation Note:
        Defaults to 'loocv' as it provides a more rigorous and 'brutal' test
        for small experimental machining datasets, reducing the chance of
        reporting 'lucky' split results.
        """
        best_r2 = -float("inf")
        best_params = {}
        best_metrics = np.zeros(3)

        keys, values = zip(*param_grid.items())
        combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        print(f"Starting Grid Search: Testing {len(combinations)} combinations...\n")
        if split == "loocv":
            print("Using Leave-One-Out cross-validator")
        else:
            print("Using K-Fold cross-validator.")
        for params in combinations:
            if split == "loocv":
                (rmse, mae, r2), preds, acts = self.run_loocv(
                    hidden_layers=params["hidden_layers"],
                    activation=params["activation"],
                    alpha=params["alpha"],
                )
            else:
                (rmse, mae, r2), preds, acts = self.run_kfold(
                    hidden_layers=params["hidden_layers"],
                    activation=params["activation"],
                    alpha=params["alpha"],
                )

            print(f"Tested: {params} | R2: {r2:.4f}, RMSE: {rmse:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                best_params = params
                best_metrics = (rmse, mae, r2)
                self.best_predictions = preds
                self.best_actuals = acts

        print("\n" + "=" * 40)
        print("BEST ARCHITECTURE FOUND ")
        print("=" * 40)
        print(f"Parameters: {best_params}")
        print(f"Overall RMSE: {best_metrics[0]:.4f}")
        print(f"Overall MAE:  {best_metrics[1]:.4f}")
        print(f"Overall R2:   {best_metrics[2]:.4f}")

        return best_params, best_metrics

    def run_loocv(self, hidden_layers, activation, alpha):
        all_predictions: list[float] = []
        all_actuals: list[float] = []

        for train_idx, test_idx in self.loo.split(self.X, self.y):
            X_train, y_train = self.X.iloc[train_idx], self.y.iloc[train_idx]
            X_test, y_test = self.X.iloc[test_idx], self.y.iloc[test_idx]

            model = NeuralNetwork(
                hidden_layers=hidden_layers, activation=activation, alpha=alpha
            )
            model.fit(X_train, y_train)

            prediction = model.predict(X_test)

            all_predictions.append(prediction)
            all_actuals.append(y_test.values[0])

        return evaluate(all_predictions, all_actuals), all_predictions, all_actuals

    def run_kfold(self, hidden_layers, activation, alpha, n_splits=5):
        """Runs the K-Fold process for one specific model configuration."""
        all_predictions = []
        all_actuals = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(self.X, self.y):
            X_train, y_train = self.X.iloc[train_idx], self.y.iloc[train_idx]
            X_test, y_test = self.X.iloc[test_idx], self.y.iloc[test_idx]

            model = NeuralNetwork(
                hidden_layers=hidden_layers, activation=activation, alpha=alpha
            )
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            all_predictions.append(predictions)
            all_actuals.append(y_test.values)

        return evaluate(all_predictions, all_actuals), all_predictions, all_actuals

    def save_plot(self, filepath: str, title="Actual vs Predicted"):
        """
        Archives a visualization of the best model's performance to disk.

        Operational Note:
        Optimized for automated reporting pipelines where GUI display is not required.
        """
        if not self.best_predictions or not self.best_actuals:
            print("Error: No results found to plot. Run grid_search first.")
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        plt.figure(figsize=(10, 6))

        plt.plot(
            self.best_actuals, label="Actual", color="blue", marker="o", linestyle="-"
        )

        plt.plot(
            self.best_predictions,
            label="Predicted",
            color="red",
            marker="x",
            linestyle="--",
        )

        plt.title(title)
        plt.xlabel("Observation Index")
        plt.ylabel("Target/Prediction Value")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()

        plt.savefig(filepath)
        plt.close()
        print(f"Plot saved to: {filepath}")

    def save_results_csv(self, filepath: str):
        """
        Exports simulation raw data to a CSV formatted for Microsoft Excel.

        Reporting Note:
        Appends 'Absolute_Error' and 'Squared_Error' columns automatically.
        Essential for auditing the model performance against specific
        machining runs.
        """
        if not self.best_predictions or not self.best_actuals:
            print("Error: No results found to save. Run grid_search first.")
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        results_df = pd.DataFrame(
            {
                "Observation": range(1, len(self.best_actuals) + 1),
                "Actual": self.best_actuals,
                "Predicted": self.best_predictions,
            }
        )

        results_df["Absolute_Error"] = (
            results_df["Actual"] - results_df["Predicted"]
        ).abs()
        results_df["Squared_Error"] = (
            results_df["Actual"] - results_df["Predicted"]
        ) ** 2

        results_df.to_csv(filepath, index=False)
        print(f"CSV results saved to: {filepath}")
