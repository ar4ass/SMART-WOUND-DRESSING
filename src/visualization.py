from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_color_comparison(
    original_rgb: tuple, analyzed_rgb: tuple, reference_rgb: Optional[tuple] = None
) -> plt.Figure:
    swatches = [("Original", original_rgb), ("Analyzed", analyzed_rgb)]
    if reference_rgb is not None:
        swatches.append(("Reference", reference_rgb))

    fig, axes = plt.subplots(1, len(swatches), figsize=(3 * len(swatches), 2.2))
    if len(swatches) == 1:
        axes = [axes]

    for ax, (label, rgb) in zip(axes, swatches):
        normalized = tuple(c / 255 for c in rgb)
        ax.add_patch(plt.Rectangle((0, 0), 1, 1, color=normalized))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f"{label}\nRGB{rgb}", fontsize=9)
        ax.axis("off")

    fig.tight_layout()
    return fig


def plot_channel_bar(values: Sequence[float], labels: Sequence[str], title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 3))
    colors = ["#d62728", "#2ca02c", "#1f77b4"] if labels[0] in ("R", "H") else None
    ax.bar(labels, values, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Nilai rata-rata")
    fig.tight_layout()
    return fig


def plot_actual_vs_predicted(actual: np.ndarray, predicted: np.ndarray, title: str = "Actual vs Predicted") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4.5, 4.5))
    ax.scatter(actual, predicted, alpha=0.6, edgecolor="k", linewidth=0.3)

    min_val = min(np.min(actual), np.min(predicted))
    max_val = max(np.max(actual), np.max(predicted))
    ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1, label="y = x (ideal)")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig
