from __future__ import annotations

from typing import Optional

import numpy as np

from src.color_analysis import ColorStats, compute_color_stats


def normalized_rgb(stats: ColorStats) -> dict:
    total = stats.mean_r + stats.mean_g + stats.mean_b
    if total == 0:
        return {"norm_r": 0.0, "norm_g": 0.0, "norm_b": 0.0}
    return {
        "norm_r": stats.mean_r / total,
        "norm_g": stats.mean_g / total,
        "norm_b": stats.mean_b / total,
    }


def extract_features(array: np.ndarray, baseline_array: Optional[np.ndarray] = None) -> dict:
    stats = compute_color_stats(array)
    features = stats.as_dict()
    features.update(normalized_rgb(stats))

    if baseline_array is not None:
        baseline_stats = compute_color_stats(baseline_array)
        features.update(
            {
                "delta_r": stats.mean_r - baseline_stats.mean_r,
                "delta_g": stats.mean_g - baseline_stats.mean_g,
                "delta_b": stats.mean_b - baseline_stats.mean_b,
                "delta_h": stats.mean_h - baseline_stats.mean_h,
                "delta_s": stats.mean_s - baseline_stats.mean_s,
                "delta_v": stats.mean_v - baseline_stats.mean_v,
            }
        )

    return features
