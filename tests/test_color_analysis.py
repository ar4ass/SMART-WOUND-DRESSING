import numpy as np
import pytest

from src.color_analysis import compute_color_stats, rgb_to_hsv


def test_pure_red_image_has_correct_rgb_mean():
    array = np.zeros((50, 50, 3), dtype=np.uint8)
    array[:, :, 0] = 255

    stats = compute_color_stats(array)

    assert stats.mean_r == pytest.approx(255)
    assert stats.mean_g == pytest.approx(0)
    assert stats.mean_b == pytest.approx(0)
    assert stats.std_r == pytest.approx(0)


def test_solid_color_hsv_matches_known_opencv_value():
    array = np.zeros((10, 10, 3), dtype=np.uint8)
    array[:, :, 0] = 255

    hsv = rgb_to_hsv(array)

    assert hsv[0, 0, 0] == 0
    assert hsv[0, 0, 1] == 255
    assert hsv[0, 0, 2] == 255


def test_stats_dict_has_all_expected_keys():
    array = np.random.randint(0, 255, (30, 30, 3), dtype=np.uint8)
    stats = compute_color_stats(array).as_dict()

    expected_keys = {
        "mean_r", "mean_g", "mean_b", "std_r", "std_g", "std_b",
        "mean_h", "mean_s", "mean_v", "std_h", "std_s", "std_v",
    }
    assert expected_keys.issubset(stats.keys())


def test_invalid_shape_raises_value_error():
    array = np.zeros((50, 50), dtype=np.uint8)
    with pytest.raises(ValueError):
        compute_color_stats(array)
