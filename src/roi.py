from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class ROIError(Exception):
    pass


@dataclass
class ROIResult:
    roi_array: np.ndarray
    x: int
    y: int
    width: int
    height: int


MIN_ROI_SIZE_PX = 20


def get_center_roi(array: np.ndarray, fraction: float = 0.5) -> ROIResult:
    if not (0 < fraction <= 1):
        raise ROIError("fraction harus di antara 0 (tidak termasuk) dan 1.")

    height, width = array.shape[:2]
    side = int(min(height, width) * fraction)
    side = max(side, MIN_ROI_SIZE_PX)

    if side > height or side > width:
        raise ROIError(
            f"ROI hasil perhitungan ({side}px) lebih besar dari gambar ({width}x{height}px)."
        )

    x = (width - side) // 2
    y = (height - side) // 2

    roi_array = array[y : y + side, x : x + side]
    return ROIResult(roi_array=roi_array, x=x, y=y, width=side, height=side)


def get_fixed_roi(array: np.ndarray, x: int, y: int, width: int, height: int) -> ROIResult:
    img_height, img_width = array.shape[:2]

    if width < MIN_ROI_SIZE_PX or height < MIN_ROI_SIZE_PX:
        raise ROIError(f"ROI terlalu kecil. Minimal {MIN_ROI_SIZE_PX}x{MIN_ROI_SIZE_PX}px.")

    if x < 0 or y < 0 or x + width > img_width or y + height > img_height:
        raise ROIError(
            f"ROI ({x},{y},{width}x{height}) berada di luar batas gambar "
            f"({img_width}x{img_height}px)."
        )

    roi_array = array[y : y + height, x : x + width]
    return ROIResult(roi_array=roi_array, x=x, y=y, width=width, height=height)
