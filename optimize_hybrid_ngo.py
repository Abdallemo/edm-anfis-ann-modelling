import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mealpy import NGO, FloatVar

from anfis_ngo import AnfisNetNGO
from ann_ngo import NeuralNetworkNGO

DATASET_PATH = "datasets/dataset2.csv"
FEATURES = ["ton", "duty_cycle", "peak_current", "voltage"]
TARGET = "surface_roughness"
RESULTS_DIR = os.path.join("results", "NGO Results")


def get_physical_bounds(csv_path: str) -> tuple[list[float], list[float], pd.DataFrame]:
    """Extracts true physical machine limits directly from the experimental dataset."""
    df = pd.read_csv(csv_path)
    lower_bounds = df[FEATURES].min().tolist()
    upper_bounds = df[FEATURES].max().tolist()
    return lower_bounds, upper_bounds, df


def load_model(model_type: str):
    """Loads the specified trained model architecture."""
    model_path = f"models/dataset2_{model_type}_NGO_model_CPU.pkl"
    print(f"\nLoading trained {model_type} model from {model_path}...")

    if model_type == "ANFIS":
        return AnfisNetNGO.load(model_path)
    elif model_type == "ANN":
        return NeuralNetworkNGO.load(model_path)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def run_ngo_optimization(
    model, lower: list[float], upper: list[float], epochs=100, pop_size=50
):
    """Executes the NGO algorithm to find optimal EDM parameters."""
    print(f"\nRunning Optimization to minimize {TARGET}...")

    def objective_function(solution):
        input_data = pd.DataFrame([solution], columns=FEATURES)
        return model.predict(input_data)

    problem = {
        "bounds": FloatVar(lb=lower, ub=upper, name="edm_vars"),
        "obj_func": objective_function,
        "minmax": "min",
    }

    solver = NGO.OriginalNGO(epoch=epochs, pop_size=pop_size)

    start_time = time.time()
    best_agent = solver.solve(problem)
    exec_time = time.time() - start_time

    return best_agent, exec_time, solver


def save_excel_report(solver, best_agent, exec_time: float, model_type: str):
    """Extracts MEALPY history and saves all optimization data to Excel."""
    excel_path = os.path.join(RESULTS_DIR, f"{model_type}_NGO_Optimization_Data.xlsx")

    epochs = len(solver.history.list_global_best_fit)
    history_df = pd.DataFrame(
        {
            "Epoch": range(1, epochs + 1),
            "Current_Best_Ra": solver.history.list_current_best_fit,
            "Global_Best_Ra": solver.history.list_global_best_fit,
            "Runtime_Seconds": solver.history.list_epoch_time,
        }
    )

    summary_data = [
        ["Total Execution Time (s)", exec_time],
        ["Predicted Minimum Ra", best_agent.target.fitness],
    ]
    for col, val in zip(FEATURES, best_agent.solution):
        summary_data.append([f"Optimal {col}", val])

    summary_df = pd.DataFrame(summary_data, columns=["Metric", "Value"])

    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        summary_df.to_excel(writer, index=False, sheet_name="Final_Results")
        history_df.to_excel(writer, index=False, sheet_name="Epoch_History")

        workbook = writer.book
        ws_summary = writer.sheets["Final_Results"]
        ws_summary.set_column(0, 0, 25)
        ws_summary.set_column(1, 1, 15)

        ws_history = writer.sheets["Epoch_History"]
        ws_history.set_column(0, 0, 10)
        ws_history.set_column(1, 2, 20)
        ws_history.set_column(3, 3, 18)

    print(f"Saved Excel Data: {excel_path}")


def plot_convergence(convergence_data: list[float], model_type: str):
    """Generates and saves the convergence behavior plot."""
    plt.figure(figsize=(8, 5))
    plt.plot(convergence_data, color="red", linewidth=2, label=f"{model_type}-NGO")
    plt.title(f"EDM Optimization Convergence Curve ({model_type}-NGO)")
    plt.xlabel("Number of Iterations")
    plt.ylabel("Predicted Surface Roughness (Ra)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, f"{model_type}_NGO_convergence.png")
    plt.savefig(out_path)
    print(f"Saved Diagram: {out_path}")
    plt.close()


def plot_surface_response(
    model, lower: list[float], upper: list[float], opt_solution, model_type: str
):
    """Generates and saves a 3D surface response map for Ton and Peak Current."""
    opt_duty = opt_solution[1]
    opt_volt = opt_solution[3]

    ton_range = np.linspace(lower[0], upper[0], 30)
    ip_range = np.linspace(lower[2], upper[2], 30)
    T, P = np.meshgrid(ton_range, ip_range)
    Z = np.zeros_like(T)

    for i in range(T.shape[0]):
        for j in range(T.shape[1]):
            row = pd.DataFrame(
                [[T[i, j], opt_duty, P[i, j], opt_volt]], columns=FEATURES
            )
            Z[i, j] = model.predict(row)

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(T, P, Z, cmap="viridis", edgecolor="none")

    ax.set_xlabel("Pulse-on Time (ton)")
    ax.set_ylabel("Peak Current")
    ax.set_zlabel("Predicted Ra")
    ax.set_title(f"Surface Response ({model_type}): Ra vs. Ton & Peak Current")
    fig.colorbar(surf, shrink=0.5, aspect=0.5, label="Surface Roughness (Ra)")

    out_path = os.path.join(RESULTS_DIR, f"{model_type}_NGO_surface_response.png")
    plt.savefig(out_path)
    print(f"Saved Diagram: {out_path}")
    plt.close()


def plot_error_analysis(model, df: pd.DataFrame, model_type: str):
    """Generates and saves the residual error analysis plot against raw data."""
    X_data = df[FEATURES]
    y_data = df[TARGET]

    predictions = [
        model.predict(pd.DataFrame([x], columns=FEATURES)) for x in X_data.values
    ]
    residuals = y_data.values - predictions

    plt.figure(figsize=(8, 5))
    plt.scatter(predictions, residuals, alpha=0.6, color="purple", edgecolors="k")
    plt.axhline(0, color="red", linestyle="--", linewidth=2)
    plt.title(f"Error Analysis (Residuals) - {model_type}-NGO")
    plt.xlabel("Predicted Surface Roughness (Ra)")
    plt.ylabel("Residual Error (Actual - Predicted)")
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.tight_layout()

    out_path = os.path.join(RESULTS_DIR, f"{model_type}_NGO_error_analysis.png")
    plt.savefig(out_path)
    print(f"Saved Diagram: {out_path}")
    plt.close()


def main():
    if len(sys.argv) < 2 or sys.argv[1].lower() not in ["ann", "anfis"]:
        print("Usage: uv run optimize_hybrid_ngo.py [ann|anfis]")
        sys.exit(1)

    model_type = sys.argv[1].upper()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    lower_bounds, upper_bounds, df_raw = get_physical_bounds(DATASET_PATH)
    print("--- Extracted Physical Boundaries ---")
    for col, lb, ub in zip(FEATURES, lower_bounds, upper_bounds):
        print(f"  {col}: [{lb} - {ub}]")

    model = load_model(model_type)

    best_agent, exec_time, solver = run_ngo_optimization(
        model=model, lower=lower_bounds, upper=upper_bounds
    )

    print("\n=============================================")
    print(f"          {model_type}-NGO RESULTS")
    print("=============================================")
    print("Optimal EDM Machining Parameters:")
    for col, val in zip(FEATURES, best_agent.solution):
        print(f"  {col}: {val:.4f}")

    print(f"\nPredicted Minimum Ra: {best_agent.target.fitness:.4f}")
    print(f"Time Execution:       {exec_time:.4f} seconds")
    print("=============================================\n")

    print("Generating Excel Report and Output Diagrams...")
    save_excel_report(solver, best_agent, exec_time, model_type)
    plot_convergence(solver.history.list_global_best_fit, model_type)
    plot_surface_response(
        model, lower_bounds, upper_bounds, best_agent.solution, model_type
    )
    plot_error_analysis(model, df_raw, model_type)

    print("\nProcess Complete.")


if __name__ == "__main__":
    main()
