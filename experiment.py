import itertools
import os
from typing import Any, Dict, Literal, cast

import matplotlib
from joblib import Parallel, delayed

import anfis
import ann
from metrics import evaluate

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold, LeaveOneOut

ParamConfig = Dict[str, Any]
ParamGrid = Dict[str, list[Any]]


class ExperimentRunner:
    """
    ExperimentRunner is the unified orchestration layer for model validation and
    hyperparameter optimization. It dynamically supports any model architecture
    (ANN or ANFIS) passed during initialization.

    Operational Requirements:
    Requires the 'evaluate' helper function to compute RMSE, MAE, and R2 metrics.
    Designed to compare two primary validation strategies:
    1. LOOCV: Leave-One-Out Cross-Validation. Recommended for small datasets.
    2. K-Fold: Standard cross-validation. Useful for verifying model stability.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_class: ann.NeuralNetwork | anfis.AnfisNet,
        device: str | None = None,
        use_polynomial: bool = True,
    ):
        """
        Args:
            X: Input machining parameters (Volt, Ip, Ton, etc.).
            y: Target surface roughness values (Ra).
            model_class: The uninstantiated class of the model to use
                         (e.g., NeuralNetwork or AnfisNet).
        """
        self.X: pd.DataFrame = X
        self.y: pd.Series = y
        self.model_class = model_class
        self.device = device
        self.use_polynomial = use_polynomial
        self.loo: LeaveOneOut = LeaveOneOut()

        self.best_predictions: list[float] = []
        self.best_actuals: list[float] = []
        self.best_result: ParamConfig | None = None
        self.best_metrics: tuple[float, float, float] | None = None

    def grid_search(
        self,
        param_grid: ParamGrid,
        split: Literal["loocv", "kfold"] = "loocv",
        n_jobs: int | None = None,
    ):
        if n_jobs is None:
            n_jobs = 1 if self.device == "gpu" else -1

        best_r2 = -float("inf")
        best_params: ParamConfig | None = None
        best_metrics: tuple[float, float, float] | None = None

        keys, values = zip(*param_grid.items())
        combinations: list[ParamConfig] = [
            cast(ParamConfig, cast(object, dict(zip(keys, v))))
            for v in itertools.product(*values)
        ]

        print(f"Starting Grid Search: Testing {len(combinations)} combinations...\n")

        def evaluate_combination(params: ParamConfig):
            if split == "loocv":
                metrics, preds, acts = self.run_loocv(params)
            else:
                metrics, preds, acts = self.run_kfold(params)
            return params, metrics, preds, acts

        parallel_executor = Parallel(n_jobs=n_jobs, return_as="generator")
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
        print(f"BEST {self.model_class.__name__} ARCHITECTURE FOUND ")
        print("=" * 40)
        if best_metrics is not None:
            print(f"Parameters: {best_params}")
            print(f"Overall RMSE: {best_metrics[0]:.4f}")
            print(f"Overall MAE:  {best_metrics[1]:.4f}")
            print(f"Overall R2:   {best_metrics[2]:.4f}")

        return best_params, best_metrics

    def run_loocv(self, params: ParamConfig):
        all_predictions: list[float] = []
        all_actuals: list[float] = []

        for train_idx, test_idx in self.loo.split(self.X, self.y):
            X_train, y_train = (
                cast(pd.DataFrame, self.X.iloc[train_idx]),
                cast(pd.Series, self.y.iloc[train_idx]),
            )
            X_test, y_test = (
                cast(pd.DataFrame, self.X.iloc[test_idx]),
                cast(pd.Series, self.y.iloc[test_idx]),
            )

            model = self.model_class(
                **params, device=self.device, use_polynomial=self.use_polynomial
            )
            model.fit(X_train, y_train)

            prediction = model.predict(X_test)

            all_predictions.append(prediction)
            all_actuals.append(y_test.values[0])

        return evaluate(all_predictions, all_actuals), all_predictions, all_actuals

    def run_kfold(self, params: ParamConfig, n_splits: int = 5):
        all_predictions = []
        all_actuals = []
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

        for train_idx, test_idx in kf.split(self.X, self.y):
            X_train, y_train = (
                cast(pd.DataFrame, self.X.iloc[train_idx]),
                cast(pd.Series, self.y.iloc[train_idx]),
            )
            X_test, y_test = (
                cast(pd.DataFrame, self.X.iloc[test_idx]),
                cast(pd.Series, self.y.iloc[test_idx]),
            )

            model = self.model_class(**params, device=self.device)
            model.fit(X_train, y_train)

            for i in range(len(X_test)):
                single_row_X = X_test.iloc[[i]]
                prediction = model.predict(single_row_X)
                all_predictions.append(prediction)
                all_actuals.append(y_test.values[i])

        return evaluate(all_predictions, all_actuals), all_predictions, all_actuals

    def save_plot(self, filepath: str, title="Actual vs Predicted"):
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
        headings = cast(pd.Series, df.loc[:, "Actual"] - df.loc[:, "Predicted"])

        df["Absolute_Error"] = headings.abs()
        df["Squared_Error"] = headings**2

        with open(filepath, "a"):
            pass

        summary_data = [[str(k), str(v)] for k, v in self.best_result.items()]
        summary_data.extend(
            [
                ["avg_rmse", self.best_metrics[0]],
                ["avg_mae", self.best_metrics[1]],
                ["r2", self.best_metrics[2]],
            ]
        )

        summary_df = pd.DataFrame(summary_data, columns=["Metric", "Value"])

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
