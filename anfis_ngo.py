from typing import Self

import joblib
import pandas as pd
import torch
import torch.nn as nn
from mealpy import NGO, FloatVar
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import StandardScaler

from model_types import ModelState


class SugenoFuzzyCore(nn.Module):
    """
    A PyTorch module representing a 5-layer Sugeno Adaptive Neuro-Fuzzy Inference System (ANFIS).

    This module implements continuous fuzzy logic through Gaussian membership functions
    and exact linear consequence evaluation via Least Squares Estimation (LSE).

    """

    def __init__(self, num_inputs: int, num_rules: int):
        """
        Args:
            num_inputs (int): The number of input features.
            num_rules (int): The number of fuzzy rules to generate.
        """
        super().__init__()
        self.num_rules = num_rules

        self.bell_centers = nn.Parameter(torch.randn(num_rules, num_inputs))
        self.bell_widths = nn.Parameter(torch.ones(num_rules, num_inputs))

        self.equation_weights = nn.Parameter(
            torch.randn(num_rules, num_inputs), requires_grad=False
        )
        self.equation_intercepts = nn.Parameter(
            torch.randn(num_rules, 1), requires_grad=False
        )

    def _gaussian_fuzzification(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the membership degree for each input across all Gaussian bell curves.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_inputs).

        Returns:
            torch.Tensor: Fuzzified tensor of shape (batch_size, num_rules, num_inputs).
        """
        x_expand = x.unsqueeze(1).expand(-1, self.num_rules, -1)
        return torch.exp(
            -0.5
            * torch.pow((x_expand - self.bell_centers) / (self.bell_widths + 1e-8), 2)
        )

    def _rule_activation(self, fuzzified_values: torch.Tensor) -> torch.Tensor:
        """
        Applies the fuzzy AND logic using the product T-norm to determine rule activation.

        Args:
            fuzzified_values (torch.Tensor): Tensor of shape (batch_size, num_rules, num_inputs).

        Returns:
            torch.Tensor: Activation weights of shape (batch_size, num_rules).
        """
        return torch.prod(fuzzified_values, dim=2)

    def _normalize_activations(self, activation_weights: torch.Tensor) -> torch.Tensor:
        """
        Normalizes activation weights across all rules to sum to 1.0.

        Args:
            activation_weights (torch.Tensor): Tensor of shape (batch_size, num_rules).

        Returns:
            torch.Tensor: Normalized weights of shape (batch_size, num_rules).
        """
        total_activation = torch.sum(activation_weights, dim=1, keepdim=True) + 1e-8
        return activation_weights / total_activation

    def _consequent_equations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calculates the linear consequent output for each rule.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_inputs).

        Returns:
            torch.Tensor: Linear outputs of shape (batch_size, num_rules).
        """
        return torch.matmul(x, self.equation_weights.t()) + self.equation_intercepts.t()

    def _aggregate_output(
        self, normalized_weights: torch.Tensor, linear_outputs: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregates the final prediction by weighting linear outputs.

        Args:
            normalized_weights (torch.Tensor): Tensor of shape (batch_size, num_rules).
            linear_outputs (torch.Tensor): Tensor of shape (batch_size, num_rules).

        Returns:
            torch.Tensor: Final predicted values of shape (batch_size,).
        """
        return torch.sum(normalized_weights * linear_outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass through the 5-layer ANFIS architecture.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_inputs).

        Returns:
            torch.Tensor: Predicted output tensor of shape (batch_size,).
        """
        l1_fuzzified = self._gaussian_fuzzification(x)
        l2_activations = self._rule_activation(l1_fuzzified)
        l3_normalized = self._normalize_activations(l2_activations)
        l4_linear_outs = self._consequent_equations(x)

        return self._aggregate_output(l3_normalized, l4_linear_outs)

    def _init_weighted_inputs(
        self, x: torch.Tensor, normalized_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Prepares the weighted input design matrix for Least Squares Estimation (LSE).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_inputs).
            normalized_weights (torch.Tensor): Tensor of shape (batch_size, num_rules).

        Returns:
            torch.Tensor: Formatted design matrix for the LSE solver.
        """
        batch_size, num_inputs = x.shape
        x_with_bias = torch.cat([x, torch.ones(batch_size, 1, device=x.device)], dim=1)

        weighted_input_matrix = torch.zeros(
            batch_size, self.num_rules * (num_inputs + 1), device=x.device
        )
        for i in range(self.num_rules):
            start = i * (num_inputs + 1)
            end = start + (num_inputs + 1)
            weighted_input_matrix[:, start:end] = (
                normalized_weights[:, i : i + 1] * x_with_bias
            )

        return weighted_input_matrix

    def _calculate_weights(
        self,
        weighted_input_matrix: torch.Tensor,
        target_values: torch.Tensor,
        alpha: float,
    ) -> torch.Tensor:
        """
        Solves the linear system using Ridge Regularization (Tikhonov regularization).

        Args:
            weighted_input_matrix (torch.Tensor): The design matrix from `_init_weighted_inputs`.
            target_values (torch.Tensor): Ground truth target tensor of shape (batch_size,).
            alpha (float): Ridge regularization penalty coefficient.

        Returns:
            torch.Tensor: Calculated optimal weights for the consequent layer.
        """
        transposed_matrix = weighted_input_matrix.t()
        identity_matrix = torch.eye(
            weighted_input_matrix.shape[1], device=weighted_input_matrix.device
        )

        safe_alpha = alpha if alpha > 0 else 1e-4

        left_side = torch.matmul(transposed_matrix, weighted_input_matrix) + (
            safe_alpha * identity_matrix
        )
        right_side = torch.matmul(transposed_matrix, target_values.unsqueeze(1))

        try:
            calculated_weights = torch.linalg.solve(left_side, right_side)
        except RuntimeError:
            calculated_weights = torch.matmul(torch.linalg.pinv(left_side), right_side)

        return calculated_weights

    def _update_equations(self, calculated_weights: torch.Tensor, num_inputs: int):
        """
        Updates the internal consequent layer parameters with the exactly calculated weights.

        Args:
            calculated_weights (torch.Tensor): The optimal weights from the LSE solver.
            num_inputs (int): The number of input features.
        """
        reshaped_weights = calculated_weights.squeeze().view(
            self.num_rules, num_inputs + 1
        )

        self.equation_weights.copy_(reshaped_weights[:, :num_inputs])
        self.equation_intercepts.copy_(reshaped_weights[:, num_inputs : num_inputs + 1])

    def calculate_linear_equations(
        self, x: torch.Tensor, y: torch.Tensor, alpha: float
    ):
        """
        Computes and applies the optimal parameters for the consequent layer using LSE.

        This method bypasses the standard computational graph to directly solve for
        the optimal linear weights, stabilizing the hybrid learning process.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_inputs).
            y (torch.Tensor): Ground truth target tensor of shape (batch_size,).
            alpha (float): Ridge regularization penalty coefficient.
        """
        with torch.no_grad():
            l1_fuzzified = self._gaussian_fuzzification(x)
            l2_activations = self._rule_activation(l1_fuzzified)
            l3_normalized = self._normalize_activations(l2_activations)

            weighted_inputs = self._init_weighted_inputs(x, l3_normalized)
            calculated_weights = self._calculate_weights(weighted_inputs, y, alpha)

            self._update_equations(calculated_weights, x.shape[1])


class AnfisNetNGO:
    """
    AnfisNet provides an encapsulated wrapper around a PyTorch-based Sugeno
    Fuzzy Inference System, mirroring the API of the NeuralNetwork class.

    Requirements:
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
    ) -> None:
        """
        Initializes the ANFIS wrapper with defined hyperparameters.

        Args:
            num_rules: The number of fuzzy rules/membership functions to generate.
            learning_rate: The step size for the LBFGS optimizer.
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

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> Self:
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

            def training_objective(weights_vector):

                split_point = self.num_rules * num_inputs
                centers = weights_vector[:split_point].reshape(
                    self.num_rules, num_inputs
                )
                widths = weights_vector[split_point:].reshape(
                    self.num_rules, num_inputs
                )

                with torch.no_grad():
                    self.model.bell_centers.copy_(
                        torch.tensor(centers, dtype=torch.float32, device=self.device)
                    )
                    self.model.bell_widths.copy_(
                        torch.tensor(widths, dtype=torch.float32, device=self.device)
                    )

                    self.model.calculate_linear_equations(
                        X_tensor, y_tensor, self.alpha
                    )

                    predictions = self.model(X_tensor)
                    criterion = nn.MSELoss()
                    loss = criterion(predictions, y_tensor).item()

                return loss

            problem_dict = {
                "bounds": FloatVar(
                    lb=(
                        [-5] * (self.num_rules * self.num_inputs)
                        + [0.01] * (self.num_rules * self.num_inputs)
                    ),
                    ub=(
                        [5] * (self.num_rules * self.num_inputs)
                        + [5] * (self.num_rules * self.num_inputs)
                    ),
                    name="anfis_weights",
                ),
                "obj_func": training_objective,
                "minmax": "min",
            }

            ngo_solver = NGO.OriginalNGO(epoch=self.epochs, pop_size=30)
            best_agent = ngo_solver.solve(problem_dict)

            split_point = self.num_rules * num_inputs
            final_centers = best_agent.solution[:split_point].reshape(
                self.num_rules, num_inputs
            )
            final_widths = best_agent.solution[split_point:].reshape(
                self.num_rules, num_inputs
            )

            with torch.no_grad():
                self.model.bell_centers.copy_(
                    torch.tensor(final_centers, dtype=torch.float32, device=self.device)
                )
                self.model.bell_widths.copy_(
                    torch.tensor(final_widths, dtype=torch.float32, device=self.device)
                )
                self.model.calculate_linear_equations(X_tensor, y_tensor, self.alpha)

            return self

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
