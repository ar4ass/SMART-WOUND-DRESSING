from __future__ import annotations

import pandas as pd

COLOR_FEATURE_COLS = ["R_mean", "G_mean", "B_mean", "H_mean", "S_mean", "V_mean"]


def repeatability_test(df: pd.DataFrame, group_cols: list[str] | None = None) -> pd.DataFrame:
    if group_cols is None:
        group_cols = ["sample_id", "lighting_condition", "smartphone_model", "distance_cm", "angle"]
    group_cols = [c for c in group_cols if c in df.columns]

    grouped = df.groupby(group_cols)[COLOR_FEATURE_COLS]
    summary = grouped.agg(["mean", "std", "count"])
    summary.columns = ["_".join(col) for col in summary.columns]
    return summary.reset_index()


def lighting_variation_test(df: pd.DataFrame) -> pd.DataFrame:
    return _variation_by(df, "lighting_condition")


def smartphone_variation_test(df: pd.DataFrame) -> pd.DataFrame:
    return _variation_by(df, "smartphone_model")


def distance_variation_test(df: pd.DataFrame) -> pd.DataFrame:
    return _variation_by(df, "distance_cm")


def angle_variation_test(df: pd.DataFrame) -> pd.DataFrame:
    return _variation_by(df, "angle")


def _variation_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    if column not in df.columns:
        raise ValueError(f"Kolom '{column}' tidak ada di dataset.")
    grouped = df.groupby(column)[COLOR_FEATURE_COLS]
    summary = grouped.agg(["mean", "std", "count"])
    summary.columns = ["_".join(col) for col in summary.columns]
    return summary.reset_index()
