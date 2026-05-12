import ast
import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_research_dashboard(excel_path: str):
    if not os.path.exists(excel_path):
        print(f"Error: Could not find {excel_path}")
        return

    print(f"Extracting data from {excel_path}...")
    xls = pd.ExcelFile(excel_path)

    if "All_Combinations" not in xls.sheet_names:
        print("Error: 'All_Combinations' sheet missing. Re-run your grid search.")
        return

    df_all = pd.read_excel(xls, sheet_name="All_Combinations")
    is_anfis = "num_rules" in df_all.columns

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    ax_actual = axes[0, 0]
    ax_trend1 = axes[0, 1]
    ax_trend2 = axes[1, 0]
    ax_trend3 = axes[1, 1]

    sheet_names = [s for s in xls.sheet_names if "Rank" in s]
    if sheet_names:
        df_first = pd.read_excel(xls, sheet_name=sheet_names[0])
        ax_actual.plot(
            df_first["Observation"],
            df_first["Actual"],
            label="Actual Target",
            color="black",
            linestyle="--",
            linewidth=2.5,
            zorder=10,
        )

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple"]
        for idx, sheet in enumerate(sheet_names[:5]):
            df = pd.read_excel(xls, sheet_name=sheet)
            ax_actual.plot(
                df["Observation"],
                df["Predicted"],
                label=sheet.replace("_", " "),
                color=colors[idx % len(colors)],
                alpha=0.8,
                linewidth=1.5,
            )

        ax_actual.set_title("Actual vs Predicted (Top 5 Ranks)", fontweight="bold")
        ax_actual.set_xlabel("Observation Index")
        ax_actual.set_ylabel("Surface Roughness (Ra)")
        ax_actual.legend()
        ax_actual.grid(True, linestyle=":", alpha=0.6)

    if is_anfis:
        fig.suptitle(
            f"ANFIS Hyperparameter Analysis Dashboard\n{os.path.basename(excel_path)}",
            fontsize=18,
            fontweight="bold",
        )

        trend1_data = df_all.groupby(["epochs", "num_rules"])["R2"].max().unstack()
        trend1_data.plot(ax=ax_trend1, marker="o", linewidth=2)
        ax_trend1.set_title("Impact of Epochs on Peak R²", fontweight="bold")
        ax_trend1.set_xlabel("Epochs")
        ax_trend1.set_ylabel("Peak R² Score")
        ax_trend1.legend(title="Num Rules")
        ax_trend1.grid(True, linestyle=":", alpha=0.6)

        trend2_data = (
            df_all.groupby(["learning_rate", "num_rules"])["RMSE"].min().unstack()
        )
        trend2_data.plot(ax=ax_trend2, marker="s", linewidth=2)
        ax_trend2.set_title("Impact of Learning Rate on Lowest RMSE", fontweight="bold")
        ax_trend2.set_xlabel("Learning Rate")
        ax_trend2.set_ylabel("Lowest RMSE (Error)")
        ax_trend2.legend(title="Num Rules")
        ax_trend2.grid(True, linestyle=":", alpha=0.6)

        trend3_data = df_all.groupby(["alpha", "num_rules"])["R2"].mean().unstack()
        trend3_data.plot(ax=ax_trend3, marker="^", linewidth=2)
        ax_trend3.set_title(
            "Impact of Ridge Penalty (Alpha) on Average R²", fontweight="bold"
        )
        ax_trend3.set_xlabel("Alpha (Ridge Penalty)")
        ax_trend3.set_ylabel("Average R² Score")
        ax_trend3.legend(title="Num Rules")
        ax_trend3.grid(True, linestyle=":", alpha=0.6)

    else:
        fig.suptitle(
            f"Neural Network Analysis Dashboard\n{os.path.basename(excel_path)}",
            fontsize=18,
            fontweight="bold",
        )

        def get_total_neurons(val):
            try:
                tup = ast.literal_eval(val) if isinstance(val, str) else val
                return sum(tup)
            except:
                return 0

        df_all["total_neurons"] = df_all["hidden_layers"].apply(get_total_neurons)

        trend1_data = df_all.groupby("total_neurons")["R2"].max()
        trend1_data.plot(ax=ax_trend1, color="tab:red", marker="o", linewidth=2)
        ax_trend1.set_title(
            "Network Capacity (Total Neurons) vs Peak R²", fontweight="bold"
        )
        ax_trend1.set_xlabel("Total Neurons in Hidden Layers")
        ax_trend1.set_ylabel("Peak R² Score")
        ax_trend1.grid(True, linestyle=":", alpha=0.6)

        trend2_data = df_all.groupby(["alpha", "activation"])["RMSE"].min().unstack()
        trend2_data.plot(ax=ax_trend2, marker="s", linewidth=2)
        ax_trend2.set_title(
            "L2 Regularization (Alpha) vs Lowest RMSE", fontweight="bold"
        )
        ax_trend2.set_xlabel("Alpha Parameter")
        ax_trend2.set_ylabel("Lowest RMSE (Error)")
        ax_trend2.legend(title="Activation Fx")
        ax_trend2.grid(True, linestyle=":", alpha=0.6)

        df_all.boxplot(column="R2", by="activation", ax=ax_trend3, grid=False)
        ax_trend3.set_title(
            "Stability of Activation Functions (R² Spread)", fontweight="bold"
        )
        ax_trend3.set_xlabel("Activation Function")
        ax_trend3.set_ylabel("R² Distribution")
        plt.suptitle(fig._suptitle.get_text(), fontsize=18, fontweight="bold")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_name = excel_path.replace(".xlsx", "_academic_dashboard.png")
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Academic dashboard successfully saved to: {output_name}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_research_dashboard(sys.argv[1])
    else:
        print("Please provide the path to the Excel file.")
        print(
            "Usage: uv run research_plotter.py results/dataset2_ANFIS_CPU_model_results.xlsx"
        )
