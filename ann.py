import itertools
import os
import pickle
from typing import Self

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class NeuralNetwork:
    def __init__(
        self, hidden_layer_sizes=(5,), activation="relu", max_iter=5000, alpha=0.0001
    ) -> None:

        self.scaler = StandardScaler()
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver="lbfgs",
            alpha=alpha,
            max_iter=max_iter,
            random_state=42,
        )

    def fit(self, X_train, y_train) -> None:
        try:
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.model.fit(X_train_scaled, y_train)
        except ValueError as e:
            raise ValueError(
                f"Training failed due to invalid data format: {e}"
            ) from None

    def predict(self, x_test):
        try:
            x_test_scaled = self.scaler.transform(x_test)
            return self.model.predict(x_test_scaled)
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
            state = {"model": self.model, "scaler": self.scaler}
            joblib.dump(state, filepath)

            print(f"Network successfully saved to {filepath}")
        except OSError as e:
            raise OSError(
                f"Could not save the model to '{filepath}'. Check directory permissions: {e}"
            ) from None

    @classmethod
    def load(cls, filepath: str) -> Self:
        """Loads a saved network from the hard drive."""
        try:
            state = joblib.load(filepath)

            if "model" not in state or "scaler" not in state:
                raise KeyError(
                    "The loaded file is missing the 'model' or 'scaler' state."
                )

            network = cls()
            network.model = state["model"]
            network.scaler = state["scaler"]

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


def evaluate(predictions, y_true):
    mse = mean_squared_error(y_true, predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, predictions)
    r2 = r2_score(y_true, predictions)
    return rmse, mae, r2


class ExperimentRunner:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.loo = LeaveOneOut()
        self.best_predictions = []
        self.best_actuals = []

    def run_loocv(self, hidden_layers, activation, alpha):
        all_predictions = []
        all_actuals = []

        for train_idx, test_idx in self.loo.split(self.X, self.y):
            X_train, y_train = self.X.iloc[train_idx], self.y.iloc[train_idx]
            X_test, y_test = self.X.iloc[test_idx], self.y.iloc[test_idx]

            model = NeuralNetwork(
                hidden_layer_sizes=hidden_layers, activation=activation, alpha=alpha
            )
            model.fit(X_train, y_train)

            prediction = model.predict(X_test)

            all_predictions.append(prediction[0])
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
                hidden_layer_sizes=hidden_layers, activation=activation, alpha=alpha
            )
            model.fit(X_train, y_train)

            predictions = model.predict(X_test)

            all_predictions.extend(predictions)
            all_actuals.extend(y_test.values)

        return evaluate(all_predictions, all_actuals), all_predictions, all_actuals

    def save_plot(self, filepath: str, title="Actual vs Predicted"):
        """
        Saves the line plot to a file without calling plt.show().
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
        Saves the actual vs predicted data into a CSV file for Excel.
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

    def grid_search(self, param_grid, split="loocv"):
        """Tests all combinations of parameters and finds the best one.
        \n`split`: takes eaither loocv or kfold which are Leave-One-Out cross-validator and K-Fold cross-validator respectfully.
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
