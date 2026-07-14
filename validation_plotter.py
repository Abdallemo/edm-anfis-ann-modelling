import os
import sys

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def generate_regression_and_error_plots(excel_path: str):
    if not os.path.exists(excel_path):
        print(f"Error: Could not find {excel_path}")
        return

    print(f"Extracting data from {excel_path}...")
    xls = pd.ExcelFile(excel_path)

    sheet_names = [s for s in xls.sheet_names if "Rank" in s]
    if not sheet_names:
        print("Error: No 'Rank' sheets found in the Excel file.")
        return

    best_sheet = sheet_names[0]
    df = pd.read_excel(xls, sheet_name=best_sheet)

    actual = df["Actual"]
    predicted = df["Predicted"]

    errors = actual - predicted

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    ax_reg = axes[0]
    ax_err = axes[1]

    ax_reg.scatter(
        actual,
        predicted,
        color="tab:blue",
        alpha=0.7,
        edgecolors="black",
        s=60,
        zorder=3,
    )

    min_val = min(actual.min(), predicted.min())
    max_val = max(actual.max(), predicted.max())
    ax_reg.plot(
        [min_val, max_val],
        [min_val, max_val],
        color="red",
        linestyle="--",
        linewidth=2.5,
        label="Ideal Fit (Y=X)",
        zorder=2,
    )

    ax_reg.set_title(f"Regression Plot ({best_sheet})", fontweight="bold")
    ax_reg.set_xlabel("Actual Surface Roughness (Ra)")
    ax_reg.set_ylabel("Predicted Surface Roughness (Ra)")
    ax_reg.legend()
    ax_reg.grid(True, linestyle=":", alpha=0.6, zorder=1)

    ax_err.hist(
        errors, bins=10, color="tab:orange", edgecolor="black", alpha=0.7, zorder=3
    )

    ax_err.axvline(
        x=0, color="red", linestyle="--", linewidth=2.5, label="Zero Error", zorder=4
    )

    ax_err.set_title("Error Distribution (Residuals)", fontweight="bold")
    ax_err.set_xlabel("Error (Actual - Predicted)")
    ax_err.set_ylabel("Frequency")
    ax_err.legend()
    ax_err.grid(True, linestyle=":", alpha=0.6, zorder=1)

    fig.suptitle(
        f"Model Validation Analysis\n{os.path.basename(excel_path)}",
        fontsize=16,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    output_name = excel_path.replace(".xlsx", "_validation_plots.png")
    plt.savefig(output_name, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Validation plots successfully saved to: {output_name}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_regression_and_error_plots(sys.argv[1])
    else:
        print("Please provide the path to the Excel file.")
        print(
            "Usage: uv run validation_plotter.py results/dataset1_ANN_CPU_model_results.xlsx"
        )
