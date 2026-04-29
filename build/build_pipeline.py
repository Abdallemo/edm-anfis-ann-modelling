# build_pipeline.py
from typing import Literal, cast

import pandas as pd

from ann import ExperimentRunner, NeuralNetwork, ParamGrid

DatasetType = Literal["dataset1-type", "dataset2-type"]


GRID_TYPE_1: ParamGrid = {
    "hidden_layers": [(8, 4), (6, 4), (5, 3)],
    "activation": ["logistic", "tanh"],
    "alpha": [0.0001, 0.001, 0.7, 0.75, 0.77],
}

GRID_TYPE_2: ParamGrid = {
    "hidden_layers": [(5,), (8, 4), (10, 5)],
    "activation": ["relu", "logistic", "tanh"],
    "alpha": [0.001, 0.01, 0.1, 0.3, 0.5],
}


FEATURES_TYPE_1 = ["volt", "ip", "ton", "toff"]
FEATURES_TYPE_2 = ["ton", "duty_cycle", "peak_current", "voltage"]


def build_and_save_model(csv_path: str, model_name: str, dataset_type: DatasetType):
    print(f"\n--- Analyzing and Building {model_name} as {dataset_type} ---")
    df = pd.read_csv(csv_path)

    csv_columns = set(df.columns)

    if dataset_type == "dataset1-type":
        if not set(FEATURES_TYPE_1).issubset(csv_columns):
            raise ValueError(
                f"Safety Check Failed! You requested '{dataset_type}', but "
                f"the CSV '{csv_path}' is missing required columns: {FEATURES_TYPE_1}"
            )
        feature_cols = FEATURES_TYPE_1
        target_col = "ra"
        grid = GRID_TYPE_1

    elif dataset_type == "dataset2-type":
        if not set(FEATURES_TYPE_2).issubset(csv_columns):
            raise ValueError(
                f"You requested '{dataset_type}', but "
                f"the CSV '{csv_path}' is missing required columns: {FEATURES_TYPE_2}"
            )
        feature_cols = FEATURES_TYPE_2
        target_col = "surface_roughness"
        grid = GRID_TYPE_2

    else:
        raise ValueError(f"Unrecognized dataset_type: {dataset_type}")

    X = cast(pd.DataFrame, df[feature_cols])
    y = cast(pd.Series, df[target_col])

    runner = ExperimentRunner(X, y)
    best_params, best_results = runner.grid_search(grid, split="loocv")
    print(best_params)
    runner.save_plot(
        f"results/{model_name}_plot.png", title=f"{model_name}: Actual vs Predicted"
    )
    runner.save_results_csv(f"results/{model_name}_results.csv")

    final_model = NeuralNetwork(**best_params)
    final_model.fit(X, y)
    final_model.save(f"models/{model_name}.pkl")
    print(f"Successfully built and saved {model_name}!")
