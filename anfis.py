from typing import Self

import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from model_types import ModelState


class SugenoFuzzyCore(nn.Module):
    """
    The internal PyTorch calculus engine representing the 5 layers of a Sugeno ANFIS.
    Upgraded to use MATLAB's Hybrid Learning Algorithm (LSE + Gradient Descent).
    """

    def __init__(self, num_inputs: int, num_rules: int):
        super().__init__()
        self.num_rules = num_rules

        self.bell_centers = nn.Parameter(torch.randn(num_rules, num_inputs))
        self.bell_widths = nn.Parameter(torch.ones(num_rules, num_inputs))

        # requires_grad=False because the optimizer will NEVER touch these.
        self.equation_weights = nn.Parameter(
            torch.randn(num_rules, num_inputs), requires_grad=False
        )
        self.equation_intercepts = nn.Parameter(
            torch.randn(num_rules, 1), requires_grad=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_expand = x.unsqueeze(1).expand(-1, self.num_rules, -1)

        fuzzified = torch.exp(
            -0.5
            * torch.pow((x_expand - self.bell_centers) / (self.bell_widths + 1e-8), 2)
        )
        firing_strength = torch.prod(fuzzified, dim=2)
        total_strength = torch.sum(firing_strength, dim=1, keepdim=True) + 1e-8
        normalized_firing = firing_strength / total_strength

        rule_outputs = (
            torch.matmul(x, self.equation_weights.t()) + self.equation_intercepts.t()
        )
        final_prediction = torch.sum(normalized_firing * rule_outputs, dim=1)

        return final_prediction

    def hybrid_lse_update(self, x: torch.Tensor, y: torch.Tensor, alpha: float):
        """
        Instantly calculates the perfect linear equations for Layer 4
        using Least Squares Estimation (LSE) with Ridge Regularization.
        """
        with torch.no_grad():
            x_expand = x.unsqueeze(1).expand(-1, self.num_rules, -1)
            fuzzified = torch.exp(
                -0.5
                * torch.pow(
                    (x_expand - self.bell_centers) / (self.bell_widths + 1e-8), 2
                )
            )
            firing_strength = torch.prod(fuzzified, dim=2)
            normalized_firing = firing_strength / (
                torch.sum(firing_strength, dim=1, keepdim=True) + 1e-8
            )

            batch_size, num_inputs = x.shape

            x_with_bias = torch.cat(
                [x, torch.ones(batch_size, 1, device=x.device)], dim=1
            )

            A = torch.zeros(
                batch_size, self.num_rules * (num_inputs + 1), device=x.device
            )
            for i in range(self.num_rules):
                start = i * (num_inputs + 1)
                end = start + (num_inputs + 1)
                A[:, start:end] = normalized_firing[:, i : i + 1] * x_with_bias

            #  Weights = (A^T * A + alpha * I)^-1 * A^T * y
            A_t = A.t()
            I = torch.eye(A.shape[1], device=x.device)
            ridge_alpha = alpha if alpha > 0 else 1e-4

            left_side = torch.matmul(A_t, A) + (ridge_alpha * I)
            right_side = torch.matmul(A_t, y.unsqueeze(1))

            try:
                solution = torch.linalg.solve(left_side, right_side)
            except RuntimeError:
                solution = torch.matmul(torch.linalg.pinv(left_side), right_side)

            solution = solution.squeeze().view(self.num_rules, num_inputs + 1)
            self.equation_weights.copy_(solution[:, :num_inputs])
            self.equation_intercepts.copy_(solution[:, num_inputs : num_inputs + 1])


class AnfisNet:
    """
    AnfisNet provides an encapsulated wrapper around a PyTorch-based Sugeno
    Fuzzy Inference System, mirroring the API of the NeuralNetwork class.

    Operational Requirements:
    1. Features must be passed as pandas DataFrames.
    2. Model persistence uses joblib to save the scaler, features, and the
       trained PyTorch module state.
    """

    def __init__(
        self,
        num_rules: int = 3,
        learning_rate: float = 0.01,
        epochs: int = 1000,
        alpha: float = 0.0,
        device: str | None = None,
        use_polynomial: bool = True,
    ) -> None:
        """
        Initializes the ANFIS wrapper with defined hyperparameters.

        Args:
            num_rules: The number of fuzzy rules/membership functions to generate.
            learning_rate: The step size for the Adam optimizer.
            epochs: Total number of training iterations.
        """
        self.scaler = StandardScaler()
        self.model = None
        self.learning_rate = learning_rate
        self.num_rules = num_rules
        self.epochs = epochs
        self.alpha = alpha

        if device == "gpu" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        try:
            torch.manual_seed(42)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(42)
            self.feature_names = X_train.columns.tolist()

            num_inputs = X_train.shape[1]

            self.model = SugenoFuzzyCore(
                num_inputs=num_inputs, num_rules=self.num_rules
            ).to(self.device)

            X_scaled = self.scaler.fit_transform(X_train)
            X_tensor = torch.tensor(X_scaled, dtype=torch.float32, device=self.device)
            y_tensor = torch.tensor(
                y_train.values, dtype=torch.float32, device=self.device
            )

            optimizer = optim.LBFGS(
                [self.model.bell_centers, self.model.bell_widths],
                lr=self.learning_rate,
                max_iter=100,
                line_search_fn="strong_wolfe",
            )
            criterion = nn.MSELoss()

            self.model.train()
            for _ in range(self.epochs):

                def closure():
                    optimizer.zero_grad()

                    self.model.hybrid_lse_update(X_tensor, y_tensor, self.alpha)

                    predictions = self.model(X_tensor)
                    loss = criterion(predictions, y_tensor)

                    loss.backward()
                    return loss

                optimizer.step(closure)

        except ValueError as e:
            raise ValueError(
                f"Training failed due to invalid data format: {e}"
            ) from None

    def predict(self, x_test: pd.DataFrame) -> float:
        """
        Predicts the output value for a given set of inputs.

        Raises:
            NotFittedError: If the model hasn't been trained or loaded.
            ValueError: If feature columns do not match training data.
        """
        if self.model is None:
            raise NotFittedError(
                "The model must be trained or loaded before making predictions."
            )

        try:
            x_test_scaled = self.scaler.transform(x_test)
            x_tensor = torch.tensor(
                x_test_scaled,
                dtype=torch.float32,
                device=self.device,
            )

            self.model.eval()
            with torch.no_grad():
                prediction = self.model(x_tensor)

            return float(prediction[0].cpu().item())

        except ValueError as e:
            raise ValueError(
                f"Prediction failed due to mismatched input features: {e}"
            ) from None

    def save(self, filepath: str) -> None:
        """Saves the PyTorch model, scaler, and features into a single file."""
        if self.model is None:
            raise NotFittedError("Cannot save an untrained model.")

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
    def load(cls, filepath: str, device: str | None = None) -> Self:
        """
        Reconstructs an AnfisNet instance from a saved state file.
        """
        try:
            state: ModelState = joblib.load(filepath)

            if "model" not in state or "scaler" not in state or "features" not in state:
                raise KeyError(
                    "The loaded file is missing the 'model' or 'scaler' state."
                )

            network = cls(device=device)
            network.model = state["model"].to(network.device)
            network.scaler = state["scaler"]
            network.feature_names = state.get("features", [])
            network.model.eval()

            return network

        except FileNotFoundError:
            raise FileNotFoundError(
                f"Error: The model file '{filepath}' does not exist."
            ) from None
        except KeyError as e:
            raise KeyError(
                f"Error: The file is corrupted or not a valid AnfisNet state: {e}"
            ) from None
        except Exception as e:
            raise RuntimeError(
                f"An unexpected error occurred while loading the model: {e}"
            ) from None
