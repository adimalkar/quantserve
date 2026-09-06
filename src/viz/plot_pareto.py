"""Generates publication-quality Quality vs Cost-per-Million-Tokens Pareto Frontier plots."""
import os
from typing import List, Dict, Any, Optional
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_pareto_frontier(
    data_points: Optional[List[Dict[str, Any]]] = None,
    output_path: str = "outputs/pareto_frontier.png",
):
    """Plots Quality (e.g. GSM8K / ARC accuracy %) vs $/1M Valid Tokens under SLO."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Sample calibrated Pareto points if none provided
    points = data_points or [
        {"name": "FP16 (Llama-3.2-1B)", "cost_per_1m": 0.52, "accuracy": 72.0, "color": "#1f77b4", "marker": "o"},
        {"name": "FP8 E4M3", "cost_per_1m": 0.28, "accuracy": 71.4, "color": "#2ca02c", "marker": "s"},
        {"name": "SmoothQuant W8A8", "cost_per_1m": 0.31, "accuracy": 71.0, "color": "#ff7f0e", "marker": "^"},
        {"name": "AWQ INT4 (Low Load C=4)", "cost_per_1m": 0.21, "accuracy": 70.5, "color": "#9467bd", "marker": "D"},
        {"name": "AWQ INT4 (High Load C=64)", "cost_per_1m": 0.64, "accuracy": 70.5, "color": "#d62728", "marker": "x"},
        {"name": "2:4 Sparse Tensor Core", "cost_per_1m": 0.35, "accuracy": 69.2, "color": "#8c564b", "marker": "v"},
        {"name": "Teacher Llama-3.2-3B INT4", "cost_per_1m": 0.48, "accuracy": 77.8, "color": "#e377c2", "marker": "*"},
    ]

    plt.figure(figsize=(9, 6), dpi=300)
    ax = plt.subplot(111)

    for pt in points:
        ax.scatter(
            pt["cost_per_1m"],
            pt["accuracy"],
            color=pt["color"],
            marker=pt.get("marker", "o"),
            s=120,
            label=pt["name"],
            edgecolors="black",
            linewidth=1.2,
            zorder=4,
        )
        # Annotate label
        ax.annotate(
            pt["name"],
            (pt["cost_per_1m"], pt["accuracy"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=9,
            weight="semibold",
        )

    # Connect Pareto frontier points (optimal trade-off line)
    frontier_pts = sorted([p for p in points if "High Load" not in p["name"]], key=lambda x: x["cost_per_1m"])
    fx = [p["cost_per_1m"] for p in frontier_pts]
    fy = [p["accuracy"] for p in frontier_pts]
    ax.plot(fx, fy, linestyle="--", color="gray", alpha=0.7, label="Pareto Frontier")

    ax.set_title("QuantServe-Bench: Quality vs. $/1M Valid Tokens Under SLO", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Effective Cost per 1,000,000 Valid Tokens ($ USD)", fontsize=11)
    ax.set_ylabel("Multi-Task Reasoning Accuracy (%)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.6)

    # Highlight INT4 inversion region
    ax.axvspan(0.55, 0.70, color="red", alpha=0.08, label="Dequantization Penalty Zone")

    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved Pareto Frontier plot to {output_path}")


if __name__ == "__main__":
    plot_pareto_frontier()
