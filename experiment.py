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
        self.best_params: ParamConfig | None = None
        self.best_metrics: tuple[float, float, float] | None = None

        self.all_results: list[dict] = []
        self.results_df: pd.DataFrame | None = None

    def grid_search(
        self,
        param_grid: ParamGrid,
        split: Literal["loocv", "kfold"] = "loocv",
        n_jobs: int | None = None,
    ):
        if n_jobs is None:
            n_jobs = 1 if self.device == "gpu" else -1

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
        self.all_results = []

        for raw_result in parallel_executor(tasks):
            params, (rmse, mae, r2), preds, acts = cast(
                tuple[
                    ParamConfig, tuple[float, float, float], list[float], list[float]
                ],
                raw_result,
            )

            print(f"Tested: {params} | R2: {r2:.4f}, RMSE: {rmse:.4f}")
            self.all_results.append(
                {
                    "params": params,
                    "rmse": rmse,
                    "mae": mae,
                    "r2": r2,
                    "preds": preds,
                    "acts": acts,
                }
            )
            self.all_results.sort(key=lambda x: x["r2"], reverse=True)
            flattened_records = []
            for res in self.all_results:
                row = {**res["params"]}
                row["R2"] = res["r2"]
                row["RMSE"] = res["rmse"]
                row["MAE"] = res["mae"]
                flattened_records.append(row)

            self.results_df = pd.DataFrame(flattened_records)

            if self.all_results:
                best = self.all_results[0]
                self.best_params = cast(ParamConfig, best["params"])
                self.best_metrics = (best["rmse"], best["mae"], best["r2"])
                self.best_predictions = best["preds"]
                self.best_actuals = best["acts"]

        print("\n" + "=" * 40)
        print(f"BEST {self.model_class.__name__} ARCHITECTURE FOUND ")
        print("=" * 40)
        if len(self.all_results) > 0 and self.best_params is not None:
            print(f"Parameters: {self.best_params}")
            print(f"Overall RMSE: {self.best_metrics[0]:.5f}")
            print(f"Overall MAE:  {self.best_metrics[1]:.5f}")
            print(f"Overall R2:   {self.best_metrics[2]:.5f}")

        return self.best_params, self.best_metrics

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

            if self.model_class is ann.NeuralNetwork:
                model = self.model_class(
                    **params, device=self.device, use_polynomial=self.use_polynomial
                )
            else:
                model = self.model_class(**params, device=self.device)

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

            if self.model_class is ann.NeuralNetwork:
                model = self.model_class(
                    **params, device=self.device, use_polynomial=self.use_polynomial
                )
            else:
                model = self.model_class(**params, device=self.device)

            model.fit(X_train, y_train)

            for i in range(len(X_test)):
                single_row_X = X_test.iloc[[i]]
                prediction = model.predict(single_row_X)
                all_predictions.append(prediction)
                all_actuals.append(y_test.values[i])

        return evaluate(all_predictions, all_actuals), all_predictions, all_actuals

    def save_plot(self, filepath: str):
        """
        Generates the 2x2 Academic Dashboard natively using memory,
        bypassing the need for external Excel reads.
        """
        if not self.all_results or self.results_df is None:
            print("Error: No results found to plot. Run grid_search first.")
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df_all = self.results_df.copy()
        is_anfis = "num_rules" in df_all.columns

        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        ax_actual = axes[0, 0]
        ax_trend1 = axes[0, 1]
        ax_trend2 = axes[1, 0]
        ax_trend3 = axes[1, 1]

        best_data = self.all_results[0]
        observations = range(1, len(best_data["acts"]) + 1)

        ax_actual.plot(
            observations,
            best_data["acts"],
            label="Actual Target",
            color="black",
            linestyle="--",
            linewidth=2.5,
            zorder=10,
        )

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        for idx, data in enumerate(self.all_results[:5]):
            rank_name = f"Rank {idx + 1} Best" if idx == 0 else f"Rank {idx + 1}"
            ax_actual.plot(
                observations,
                data["preds"],
                label=rank_name,
                color=colors[idx % len(colors)],
                alpha=0.8,
                linewidth=1.5,
            )

        ax_actual.set_title("Actual vs Predicted (Top 5 Ranks)", fontweight="bold")
        ax_actual.set_xlabel("Observation Index")
        ax_actual.set_ylabel("Surface Roughness (Ra)")
        ax_actual.legend()
        ax_actual.grid(True, linestyle=":", alpha=0.6)

        if is_anfis:
            fig.suptitle(
                f"ANFIS Hyperparameter Analysis Dashboard\n{os.path.basename(filepath)}",
                fontsize=18,
                fontweight="bold",
            )
            trend1_data = df_all.groupby(["epochs", "num_rules"])["R2"].max().unstack()
            trend1_data.plot(ax=ax_trend1, marker="o", linewidth=2)
            ax_trend1.set_title("Impact of Epochs on Peak R²", fontweight="bold")
            ax_trend1.set_xlabel("Epochs")
            ax_trend1.set_ylabel("Peak R² Score")
            ax_trend1.legend(title="Num Rules")
            ax_trend1.grid(True, linestyle=":", alpha=0.6)

            trend2_data = (
                df_all.groupby(["learning_rate", "num_rules"])["RMSE"].min().unstack()
            )
            trend2_data.plot(ax=ax_trend2, marker="s", linewidth=2)
            ax_trend2.set_title(
                "Impact of Learning Rate on Lowest RMSE", fontweight="bold"
            )
            ax_trend2.set_xlabel("Learning Rate")
            ax_trend2.set_ylabel("Lowest RMSE (Error)")
            ax_trend2.legend(title="Num Rules")
            ax_trend2.grid(True, linestyle=":", alpha=0.6)

            trend3_data = df_all.groupby(["alpha", "num_rules"])["R2"].mean().unstack()
            trend3_data.plot(ax=ax_trend3, marker="^", linewidth=2)
            ax_trend3.set_title(
                "Impact of Ridge Penalty (Alpha) on Average R²", fontweight="bold"
            )
            ax_trend3.set_xlabel("Alpha (Ridge Penalty)")
            ax_trend3.set_ylabel("Average R² Score")
            ax_trend3.legend(title="Num Rules")
            ax_trend3.grid(True, linestyle=":", alpha=0.6)

        else:
            fig.suptitle(
                f"Neural Network Analysis Dashboard\n{os.path.basename(filepath)}",
                fontsize=18,
                fontweight="bold",
            )

            df_all["total_neurons"] = df_all["hidden_layers"].apply(
                lambda x: sum(x) if isinstance(x, tuple) else 0
            )

            trend1_data = df_all.groupby("total_neurons")["R2"].max()
            trend1_data.plot(ax=ax_trend1, color="tab:red", marker="o", linewidth=2)
            ax_trend1.set_title(
                "Network Capacity (Total Neurons) vs Peak R²", fontweight="bold"
            )
            ax_trend1.set_xlabel("Total Neurons in Hidden Layers")
            ax_trend1.set_ylabel("Peak R² Score")
            ax_trend1.grid(True, linestyle=":", alpha=0.6)

            trend2_data = (
                df_all.groupby(["alpha", "activation"])["RMSE"].min().unstack()
            )
            trend2_data.plot(ax=ax_trend2, marker="s", linewidth=2)
            ax_trend2.set_title(
                "L2 Regularization (Alpha) vs Lowest RMSE", fontweight="bold"
            )
            ax_trend2.set_xlabel("Alpha Parameter")
            ax_trend2.set_ylabel("Lowest RMSE (Error)")
            ax_trend2.legend(title="Activation Fx")
            ax_trend2.grid(True, linestyle=":", alpha=0.6)

            df_all.boxplot(column="R2", by="activation", ax=ax_trend3, grid=False)
            ax_trend3.set_title(
                "Stability of Activation Functions (R² Spread)", fontweight="bold"
            )
            ax_trend3.set_xlabel("Activation Function")
            ax_trend3.set_ylabel("R² Distribution")
            plt.suptitle(fig._suptitle.get_text(), fontsize=18, fontweight="bold")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Academic dashboard successfully saved to: {filepath}")

    def save_results_excel(self, filepath: str):
        if not self.all_results:
            print("Error: No results found to save. Run grid_search first.")
            return

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with pd.ExcelWriter(filepath, engine="xlsxwriter") as writer:
            workbook = writer.book

            for rank in range(min(5, len(self.all_results))):
                data = self.all_results[rank]

                sheet_name = (
                    f"Rank_{rank + 1}_Best" if rank == 0 else f"Rank_{rank + 1}"
                )
                digits = decimal_places(data["acts"][0])
                df = pd.DataFrame(
                    {
                        "Observation": range(1, len(data["acts"]) + 1),
                        "Actual": data["acts"],
                        "Predicted": data["preds"],
                    }
                )
                df["Predicted"] = df["Predicted"].round(digits)
                headings = cast(pd.Series, df.loc[:, "Actual"] - df.loc[:, "Predicted"])
                df["Absolute_Error"] = headings.abs()
                df["Squared_Error"] = headings**2

                summary_data = [[str(k), str(v)] for k, v in data["params"].items()]
                summary_data.extend(
                    [
                        ["rmse", data["rmse"]],
                        ["mae", data["mae"]],
                        ["r2", data["r2"]],
                    ]
                )
                summary_df = pd.DataFrame(summary_data, columns=["Metric", "Value"])

                df.to_excel(writer, index=False, sheet_name=sheet_name)
                summary_df.to_excel(
                    writer, index=False, startcol=7, sheet_name=sheet_name
                )

                worksheet = writer.sheets[sheet_name]
                for i in range(len(df.columns)):
                    worksheet.set_column(i, i, 18)
                worksheet.set_column(7, 8, 20)

                bold = workbook.add_format({"bold": True})
                worksheet.write(0, 7, "Metric", bold)
                worksheet.write(0, 8, "Value", bold)

            if self.results_df is not None:
                self.results_df.to_excel(
                    writer, index=False, sheet_name="All_Combinations"
                )

                ws_all = writer.sheets["All_Combinations"]
                for i in range(len(self.results_df.columns)):
                    ws_all.set_column(i, i, 15)

        print(f"Results saved to: {filepath}")


def decimal_places(value: float) -> int:
    return len(str(value).split(".")[1])
