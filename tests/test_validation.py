import pandas as pd
import pytest

from src.validation import (
    angle_variation_test,
    distance_variation_test,
    lighting_variation_test,
    repeatability_test,
    smartphone_variation_test,
)


def _make_dummy_df():
    rows = []
    for lighting in ["indoor", "outdoor"]:
        for i in range(3):
            rows.append(
                {
                    "sample_id": "S001",
                    "R_mean": 150 + i, "G_mean": 100 + i, "B_mean": 90 + i,
                    "H_mean": 10, "S_mean": 100, "V_mean": 150,
                    "lighting_condition": lighting,
                    "smartphone_model": "Phone X",
                    "distance_cm": 15,
                    "angle": 90,
                }
            )
    return pd.DataFrame(rows)


def test_repeatability_test_groups_and_computes_std():
    df = _make_dummy_df()
    result = repeatability_test(df)

    assert "R_mean_std" in result.columns
    assert "R_mean_count" in result.columns
    assert (result["R_mean_count"] == 3).all()


def test_lighting_variation_test_groups_by_lighting():
    df = _make_dummy_df()
    result = lighting_variation_test(df)

    assert set(result["lighting_condition"]) == {"indoor", "outdoor"}
    assert "R_mean_mean" in result.columns


def test_smartphone_variation_test_groups_by_phone():
    df = _make_dummy_df()
    result = smartphone_variation_test(df)
    assert "smartphone_model" in result.columns


def test_distance_variation_test_groups_by_distance():
    df = _make_dummy_df()
    result = distance_variation_test(df)
    assert "distance_cm" in result.columns


def test_angle_variation_test_groups_by_angle():
    df = _make_dummy_df()
    result = angle_variation_test(df)
    assert "angle" in result.columns


def test_variation_test_missing_column_raises_error():
    df = _make_dummy_df().drop(columns=["lighting_condition"])
    with pytest.raises(ValueError):
        lighting_variation_test(df)
