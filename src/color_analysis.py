from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ColorStats:
    mean_r: float
    mean_g: float
    mean_b: float
    std_r: float
    std_g: float
    std_b: float
    mean_h: float
    mean_s: float
    mean_v: float
    std_h: float
    std_s: float
    std_v: float

    def as_dict(self) -> dict:
        return self.__dict__.copy()


def rgb_to_hsv(array: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(array, cv2.COLOR_RGB2HSV)


def compute_color_stats(array: np.ndarray) -> ColorStats:
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"Array harus berbentuk (H, W, 3). Didapat: {array.shape}")

    hsv = rgb_to_hsv(array)

    r_channel = array[:, :, 0].astype(np.float64)
    g_channel = array[:, :, 1].astype(np.float64)
    b_channel = array[:, :, 2].astype(np.float64)

    h_channel = hsv[:, :, 0].astype(np.float64)
    s_channel = hsv[:, :, 1].astype(np.float64)
    v_channel = hsv[:, :, 2].astype(np.float64)

    return ColorStats(
        mean_r=float(np.mean(r_channel)),
        mean_g=float(np.mean(g_channel)),
        mean_b=float(np.mean(b_channel)),
        std_r=float(np.std(r_channel)),
        std_g=float(np.std(g_channel)),
        std_b=float(np.std(b_channel)),
        mean_h=float(np.mean(h_channel)),
        mean_s=float(np.mean(s_channel)),
        mean_v=float(np.mean(v_channel)),
        std_h=float(np.std(h_channel)),
        std_s=float(np.std(s_channel)),
        std_v=float(np.std(v_channel)),
    )
