from typing import TypedDict

from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


class ModelState(TypedDict):
    model: MLPRegressor
    scaler: StandardScaler
    features: list[str]
