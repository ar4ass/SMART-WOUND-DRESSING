import numpy as np
import pytest

from src.roi import ROIError, get_center_roi, get_fixed_roi


def test_center_roi_is_square_and_centered():
    array = np.zeros((200, 400, 3), dtype=np.uint8)

    result = get_center_roi(array, fraction=0.5)

    assert result.width == 100
    assert result.height == 100
    assert result.x == 150
    assert result.y == 50
    assert result.roi_array.shape == (100, 100, 3)


def test_center_roi_invalid_fraction_raises_error():
    array = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ROIError):
        get_center_roi(array, fraction=0)
    with pytest.raises(ROIError):
        get_center_roi(array, fraction=1.5)


def test_fixed_roi_valid_coordinates():
    array = np.zeros((300, 300, 3), dtype=np.uint8)

    result = get_fixed_roi(array, x=10, y=20, width=50, height=60)

    assert result.roi_array.shape == (60, 50, 3)
    assert (result.x, result.y, result.width, result.height) == (10, 20, 50, 60)


def test_fixed_roi_out_of_bounds_raises_error():
    array = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ROIError):
        get_fixed_roi(array, x=80, y=80, width=50, height=50)


def test_fixed_roi_too_small_raises_error():
    array = np.zeros((100, 100, 3), dtype=np.uint8)
    with pytest.raises(ROIError):
        get_fixed_roi(array, x=0, y=0, width=5, height=5)
