import numpy as np
import pytest

from src.feature_extraction import extract_features


def test_extract_features_without_baseline():
    array = np.full((40, 40, 3), 100, dtype=np.uint8)

    features = extract_features(array)

    for key in ["mean_r", "mean_g", "mean_b", "mean_h", "mean_s", "mean_v", "norm_r", "norm_g", "norm_b"]:
        assert key in features

    assert "delta_r" not in features


def test_normalized_rgb_sums_to_one():
    array = np.zeros((20, 20, 3), dtype=np.uint8)
    array[:, :, 0] = 150
    array[:, :, 1] = 50
    array[:, :, 2] = 50

    features = extract_features(array)

    total_norm = features["norm_r"] + features["norm_g"] + features["norm_b"]
    assert total_norm == pytest.approx(1.0)


def test_extract_features_with_baseline_computes_delta():
    baseline = np.full((30, 30, 3), 100, dtype=np.uint8)
    current = np.full((30, 30, 3), 150, dtype=np.uint8)

    features = extract_features(current, baseline_array=baseline)

    assert features["delta_r"] == pytest.approx(50)
    assert features["delta_g"] == pytest.approx(50)
    assert features["delta_b"] == pytest.approx(50)


def test_extract_features_identical_baseline_gives_zero_delta():
    array = np.full((25, 25, 3), 120, dtype=np.uint8)

    features = extract_features(array, baseline_array=array)

    assert features["delta_r"] == pytest.approx(0)
    assert features["delta_h"] == pytest.approx(0)
