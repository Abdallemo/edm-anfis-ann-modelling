# build_pipeline.py
from typing import Any, Dict, Literal, Union, cast

import joblib
import pandas as pd

from anfis import AnfisNet
from ann import NeuralNetwork
from experiment import ExperimentRunner

DatasetType = Literal["dataset1-type", "dataset2-type"]
ArchitectureType = Literal["ANN", "ANFIS"]
DeviceType = Literal["cpu", "gpu"]
BuildType = Union[DeviceType | Literal["both"]]

ParamGrid = Dict[str, list[Any]]

GRID_TYPE_1: ParamGrid = {
    "hidden_layers": [
        (8, 4),
        (6, 4),
        (5, 3),
        (4, 4),
        (3, 2),
        (1, 4),
        (4, 1),
        (8, 8),
        (5, 5),
    ],
    "activation": ["logistic", "tanh"],
    "alpha": [
        0.0001,
        0.001,
        0.7,
        0.75,
        0.77,
        0.80,
        0.82,
        0.85,
        0.88,
        0.90,
        0.95,
        1.0,
        1.05,
        1.1,
    ],
}
# GRID_TYPE_1: ParamGrid = {
#     "hidden_layers": [(5, 3), (4, 2), (5,), (6,)],
#     "activation": ["logistic", "tanh"],
#     "alpha": [0.72, 0.75, 0.78, 0.80],
# }

# GRID_TYPE_1: ParamGrid = {
#     "hidden_layers": [
#         (3, 2, 1),
#         (4, 4),
#         (4, 2),
#         (3, 2),
#         (2, 3),
#         (3, 3),
#         (5, 3),  # The current baseline
#         (8, 6, 4, 2),  # 4 Hidden Layers
#         (10, 8, 6, 4, 2),  # 5 Hidden Layers
#     ],
#     "activation": ["logistic", "tanh"],
#     "alpha": [0.75, 0.78, 0.65, 0.80, 1.0, 2.0, 5.0],
# }


GRID_TYPE_2: ParamGrid = {
    "hidden_layers": [(5,), (8, 4), (10, 5), (4, 4), (5, 1), (5, 5)],
    "activation": ["relu", "logistic", "tanh"],
    "alpha": [0.001, 0.01, 0.1, 0.3, 0.5],
}
# old
# ANFIS_GRID_TYPE_1: ParamGrid = {
#     "num_rules": [2, 3, 4, 5],
#     "learning_rate": [0.1, 0.5, 1.0],
#     "epochs": [10, 20],
#     "alpha": [
#         0.6,
#         0.75,
#         0.9,
#     ],
# }


# ANFIS_GRID_TYPE_2: ParamGrid = {
#     "num_rules": [4, 5, 6],
#     "learning_rate": [0.1, 0.5, 1.0],
#     "epochs": [10, 20],
#     "alpha": [
#         0.0,
#         0.005,
#         0.01,
#     ],
# }
ANFIS_GRID_TYPE_1: ParamGrid = {
    "num_rules": [2, 3, 4, 5],
    "learning_rate": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0],
    "epochs": [10, 20, 30, 40, 50, 60],
    "alpha": [0.5, 0.7, 0.8, 0.9, 0.95, 1.0, 1.2, 1.5, 2.0],
}

# ANFIS_GRID_TYPE_2: ParamGrid = {
#     "num_rules": [4, 5, 6, 7],
#     "learning_rate": [0.1, 0.2, 0.3, 0.4],
#     "epochs": [20, 30, 40],
#     "alpha": [0.0, 0.001, 0.002, 0.005],
# }

ANFIS_GRID_TYPE_2: ParamGrid = {
    "num_rules": [2, 3, 4, 5, 6, 7],
    "learning_rate": [
        0.1,
        0.2,
        0.3,
        0.35,
        0.4,
        0.45,
        0.5,
        0.55,
    ],
    "epochs": [30, 40, 45, 60, 75, 80, 90],
    "alpha": [0.0, 0.001, 0.002, 0.005],
}

# ANFIS_GRID_TYPE_2: ParamGrid = {
#     "num_rules": [2, 4, 7],
#     "learning_rate": [
#         0.1,
#         0.5,
#     ],
#     "epochs": [30, 40, 80],
#     "alpha": [0.0, 0.005],
# }


FEATURES_TYPE_1 = ["volt", "ip", "ton", "toff"]
FEATURES_TYPE_2 = ["ton", "duty_cycle", "peak_current", "voltage"]


def build_and_save_model(
    csv_path: str,
    model_name: str,
    dataset_type: DatasetType,
    architecture: ArchitectureType = "ANN",
    run_on: DeviceType = "cpu",
    build_for: BuildType = "cpu",
):
    print(
        f"\n--- Analyzing and Building {model_name} ({architecture}) as {dataset_type} ---"
    )
    df = pd.read_csv(csv_path)
    csv_columns = set(df.columns)

    if dataset_type == "dataset1-type":
        if not set(FEATURES_TYPE_1).issubset(csv_columns):
            raise ValueError(
                f"You requested '{dataset_type}', but "
                f"the CSV '{csv_path}' is missing required columns: {FEATURES_TYPE_1}"
            )
        feature_cols = FEATURES_TYPE_1
        target_col = "ra"
        ann_grid = GRID_TYPE_1
        anfis_grid = ANFIS_GRID_TYPE_1

    elif dataset_type == "dataset2-type":
        if not set(FEATURES_TYPE_2).issubset(csv_columns):
            raise ValueError(
                f"You requested '{dataset_type}', but "
                f"the CSV '{csv_path}' is missing required columns: {FEATURES_TYPE_2}"
            )
        feature_cols = FEATURES_TYPE_2
        target_col = "surface_roughness"
        ann_grid = GRID_TYPE_2
        anfis_grid = ANFIS_GRID_TYPE_2

    else:
        raise ValueError(f"Unrecognized dataset_type: {dataset_type}")

    X = cast(pd.DataFrame, df[feature_cols])
    y = cast(pd.Series, df[target_col])

    if architecture == "ANN":
        if dataset_type == "dataset1-type":
            use_polynomial = True
        else:
            use_polynomial = False
        runner = ExperimentRunner(X, y, NeuralNetwork, run_on, use_polynomial)
        best_params, _ = runner.grid_search(ann_grid, split="loocv")

        print(best_params)

        if best_params is None:
            raise RuntimeError(
                "No best parameters found for ANN. Run grid_search first."
            )

        runner.save_plot(
            f"results/{model_name}_plot.png",
        )
        runner.save_results_excel(f"results/{model_name}_results.xlsx")

        NeuralNetwork(
            hidden_layers=best_params["hidden_layers"],
            activation=best_params["activation"],
            alpha=best_params["alpha"],
            device=run_on,
            use_polynomial=use_polynomial,
        ).fit(X, y).save(f"models/{model_name}.pkl")

    elif architecture == "ANFIS":
        runner = ExperimentRunner(X, y, AnfisNet, run_on, False)
        best_params, _ = runner.grid_search(anfis_grid, split="loocv")
        print(best_params)

        if best_params is None:
            raise RuntimeError(
                "No best parameters found for ANFIS. Run grid_search first."
            )

        runner.save_plot(
            f"results/{model_name}_plot.png",
        )
        runner.save_results_excel(f"results/{model_name}_results.xlsx")

        if build_for == "both":
            AnfisNet(
                num_rules=best_params["num_rules"],
                learning_rate=best_params["learning_rate"],
                epochs=best_params["epochs"],
                device="cpu",
            ).fit(X, y).save(f"models/{model_name}_CPU.pkl")

            AnfisNet(
                num_rules=best_params["num_rules"],
                learning_rate=best_params["learning_rate"],
                epochs=best_params["epochs"],
                device="cuda",
            ).fit(X, y).save(f"models/{model_name}_GPU.pkl")

        else:
            AnfisNet(
                num_rules=best_params["num_rules"],
                learning_rate=best_params["learning_rate"],
                epochs=best_params["epochs"],
                device=run_on,
            ).fit(X, y).save(f"models/{model_name}__{run_on}.pkl")

    else:
        raise ValueError(f"Unrecognized architecture: {architecture}")

    print(f"Successfully built and saved {model_name}!")


def load_and_train(model_path: str, csv_path: str, dataset_type: DatasetType):
    """
    Loads an existing model architecture, retrains it on new data,
    and overwrites the original file.
    """
    df = pd.read_csv(csv_path)
    csv_columns = set(df.columns)

    if dataset_type == "dataset1-type":
        if not set(FEATURES_TYPE_1).issubset(csv_columns):
            raise ValueError(
                f"You requested '{dataset_type}', but "
                f"the CSV '{csv_path}' is missing required columns: {FEATURES_TYPE_1}"
            )
        feature_cols = FEATURES_TYPE_1
        target_col = "ra"

    elif dataset_type == "dataset2-type":
        if not set(FEATURES_TYPE_2).issubset(csv_columns):
            raise ValueError(
                f"You requested '{dataset_type}', but "
                f"the CSV '{csv_path}' is missing required columns: {FEATURES_TYPE_2}"
            )
        feature_cols = FEATURES_TYPE_2
        target_col = "surface_roughness"

    else:
        raise ValueError(f"Unrecognized dataset_type: {dataset_type}")

    raw_state = joblib.load(model_path)
    engine_name = type(raw_state.get("model")).__name__

    if engine_name == "SugenoFuzzyCore":
        model = AnfisNet.load(model_path)
    elif engine_name == "MLPRegressor":
        model = NeuralNetwork.load(model_path)
    else:
        raise ValueError(f"Unrecognized internal engine: {engine_name}")

    missing_cols = set(model.feature_names) - set(feature_cols)
    if missing_cols:
        raise ValueError(
            f"Mismatch between saved model features and new dataset: {missing_cols}"
        )

    X = cast(pd.DataFrame, df[feature_cols])
    y = cast(pd.Series, df[target_col])

    model.fit(X, y).save(model_path)
