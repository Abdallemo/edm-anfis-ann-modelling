import matplotlib.pyplot as plt
import numpy as np
from mealpy import NGO, FloatVar


def sphere_function(solution):
    """
    The Sphere function: f(x) = sum(x^2).
    A standard benchmark where the perfect minimum is exactly 0.
    """
    return np.sum(solution**2)


problem_dict = {
    "bounds": FloatVar(lb=(-100.0,) * 30, ub=(100.0,) * 30, name="sphere_vars"),
    "obj_func": sphere_function,
    "minmax": "min",
}


model = NGO.OriginalNGO(epoch=100, pop_size=50)


print("Running Northern Goshawk Optimization (NGO) on Sphere Benchmark...")
best_agent = model.solve(problem_dict)

print("\n--- NGO Optimization Complete ---")
print(f"Best Solution (Optimal Parameters):\n{best_agent.solution}")
print(f"\nBest Fitness Value (Expected ~0): {best_agent.target.fitness}")


convergence_data = model.history.list_global_best_fit

plt.figure(figsize=(8, 5))
plt.plot(convergence_data, color="blue", linewidth=2)
plt.title("NGO Convergence Curve (Sphere Benchmark)")
plt.xlabel("Number of Iterations")
plt.ylabel("Best Fitness Value")
plt.grid(True)
plt.tight_layout()

plt.savefig("ngo_benchmark_convergence.png")
print("\nConvergence plot saved as 'ngo_benchmark_convergence.png'")
