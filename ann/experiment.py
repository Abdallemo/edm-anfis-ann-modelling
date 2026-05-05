import itertools
import os
from typing import Any, Literal, cast

import matplotlib
from joblib import Parallel, delayed

from ._model import NeuralNetwork
from .metrics import evaluate
from .types import ActivationType, ParamGrid

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut


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

    def __init__(self, X: pd.DataFrame, y: pd.Series):
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

        def evaluate_combination(params):
            if split == "loocv":
                metrics, preds, acts = self.run_loocv(
                    hidden_layers=params["hidden_layers"],
                    activation=params["activation"],
                    alpha=params["alpha"],
                )
            else:
                metrics, preds, acts = self.run_kfold(
                    hidden_layers=params["hidden_layers"],
                    activation=params["activation"],
                    alpha=params["alpha"],
                )
            return params, metrics, preds, acts

        parallel_executor = Parallel(n_jobs=-1, return_as="generator")
        tasks = (delayed(evaluate_combination)(params) for params in combinations)

        for raw_result in parallel_executor(tasks):
            params, (rmse, mae, r2), preds, acts = cast(
                tuple[
                    dict[str, Any], tuple[float, float, float], list[float], list[float]
                ],
                raw_result,
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

    def run_loocv(self, hidden_layers, activation: ActivationType, alpha: float):
        all_predictions: list[float] = []
        all_actuals: list[float] = []

        for train_idx, test_idx in self.loo.split(self.X, self.y):
            X_train, y_train = self.X.iloc[train_idx], self.y.iloc[train_idx]
            X_test, y_test = (
                cast(pd.DataFrame, self.X.iloc[test_idx]),
                cast(pd.Series, self.y.iloc[test_idx]),
            )

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
