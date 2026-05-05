from typing import Literal, TypedDict

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class ModelState(TypedDict):
    model: MLPRegressor
    scaler: StandardScaler
    features: list[str]


ActivationType = Literal["relu", "logistic", "tanh"]


class ParamGrid(TypedDict):
    hidden_layers: list[tuple[int, ...]]
    activation: list[ActivationType]
    """allowed Activation functions (‘logistic’, ‘tanh’, ‘relu’)."""
    alpha: list[float]
