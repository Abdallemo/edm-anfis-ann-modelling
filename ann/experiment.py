import itertools
import os
from typing import Any, Literal, cast

import matplotlib
from joblib import Parallel, delayed

from ._model import NeuralNetwork
from .metrics import evaluate
from .types import ActivationType, ParamConfig, ParamGrid

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
        self.best_result: ParamConfig | None = None
        self.best_metrics: tuple[float, float, float] | None = None

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
        best_params: ParamConfig | None = None
        best_metrics: tuple[float, float, float] | None = None

        keys, values = zip(*param_grid.items())
        combinations = [
            cast(ParamConfig, dict(zip(keys, v))) for v in itertools.product(*values)
        ]

        print(f"Starting Grid Search: Testing {len(combinations)} combinations...\n")

        def evaluate_combination(params: ParamConfig):
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
                    ParamConfig, tuple[float, float, float], list[float], list[float]
                ],
                raw_result,
            )

            print(f"Tested: {params} | R2: {r2:.4f}, RMSE: {rmse:.4f}")

            if r2 > best_r2:
                best_r2 = r2
                best_params = params
                best_metrics = (rmse, mae, r2)

                self.best_result = params
                self.best_metrics = best_metrics
                self.best_predictions = preds
                self.best_actuals = acts

        print("\n" + "=" * 40)
        print("BEST ARCHITECTURE FOUND ")
        print("=" * 40)
        if best_metrics is not None:
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
        if (
            not self.best_predictions
            or not self.best_actuals
            or not self.best_result
            or not self.best_metrics
        ):
            print("Error: No results found to save. Run grid_search first.")
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        df = pd.DataFrame(
            {
                "Observation": range(1, len(self.best_actuals) + 1),
                "Actual": self.best_actuals,
                "Predicted": self.best_predictions,
            }
        )

        df["Absolute_Error"] = (df["Actual"] - df["Predicted"]).abs()
        df["Squared_Error"] = (df["Actual"] - df["Predicted"]) ** 2

        df.to_csv(filepath, index=False)

        summary_data = [
            ["hidden_layers", str(self.best_result["hidden_layers"])],
            ["activation", self.best_result["activation"]],
            ["alpha", self.best_result["alpha"]],
            ["avg_rmse", self.best_metrics[0]],
            ["avg_mae", self.best_metrics[1]],
            ["r2", self.best_metrics[2]],
        ]

        summary_df = pd.DataFrame(summary_data, columns=["Metric", "Value"])

        with open(filepath, "a"):
            pass

        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            df.to_excel(writer, index=False, sheet_name="Results")

            summary_df.to_excel(
                writer,
                index=False,
                startcol=7,
                sheet_name="Results",
            )

            workbook = writer.book
            worksheet = writer.sheets["Results"]

            for i in range(len(df.columns)):
                worksheet.set_column(i, i, 18)

            worksheet.set_column(7, 8, 20)

            bold = workbook.add_format({"bold": True})
            worksheet.write(0, 7, "Metric", bold)
            worksheet.write(0, 8, "Value", bold)

        print(f"Results saved to: {filepath}")
