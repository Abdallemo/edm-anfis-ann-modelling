import pandas as pd

from ann import ExperimentRunner

df = pd.read_csv("./datasets/dataset2.csv")

X = df[["ton", "duty_cycle", "peak_current", "voltage"]]
y = df["surface_roughness"]


hyperparameter_grid = {
    "hidden_layers": [(5,), (8, 4), (10, 5)],
    "activation": ["relu", "logistic", "tanh"],
    "alpha": [0.001, 0.01, 0.1, 0.3, 0.5],
}

runner = ExperimentRunner(X, y)
best_parameters, best_results = runner.grid_search(hyperparameter_grid)
runner.save_plot("results/dataset2_plot.png", title="Dataset 1: Actual vs Predicted Ra")
runner.save_results_csv("results/dataset2_results.csv")
