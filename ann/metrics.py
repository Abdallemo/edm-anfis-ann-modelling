from typing import cast

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate(prediction: list[float], y_true: list[float]):
    mse = mean_squared_error(y_true, prediction)
    rmse: float = np.sqrt(mse)

    mae = cast(float, mean_absolute_error(y_true, prediction))
    r2 = cast(float, r2_score(y_true, prediction))
    return rmse, mae, r2
