# %%
import pandas as pd

from ann import ExperimentRunner

# %%
df = pd.read_csv("./datasets/dataset1.csv")
print(df.head())
X = df[["volt", "ip", "ton", "toff"]]
y = df["ra"]

# %%
hyperparameter_grid = {
    "hidden_layers": [(8, 4), (6, 4), (5, 3), (4, 2), (10, 5), (12, 6)],
    "activation": ["logistic"],
    "alpha": [0.6, 0.7, 0.75, 0.77, 0.8, 1.0, 1.1],
}


runner = ExperimentRunner(X, y)
best_parameters, best_results = runner.grid_search(hyperparameter_grid, split="kfold")
runner.save_plot(
    "results/dataset1_plot2.png", title="Dataset 1: Actual vs Predicted Ra"
)
runner.save_results_csv("results/dataset1_results2.csv")
