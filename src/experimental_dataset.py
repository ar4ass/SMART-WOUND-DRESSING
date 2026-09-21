from __future__ import annotations

import pandas as pd

EXPERIMENTAL_COLUMNS = [
    "sample_id",
    "date",
    "material_id",
    "pH_reference",
    "image_path",
    "R_mean",
    "G_mean",
    "B_mean",
    "H_mean",
    "S_mean",
    "V_mean",
    "lighting_condition",
    "smartphone_model",
    "distance_cm",
    "angle",
    "replicate",
]

DATASET_TYPE_LABEL = "experimental"


class ExperimentalDataError(Exception):
    pass


def generate_template(path: str) -> None:
    df = pd.DataFrame(columns=EXPERIMENTAL_COLUMNS)
    df.to_csv(path, index=False)


def load_experimental_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    validate_experimental_dataframe(df)
    return df


def validate_experimental_dataframe(df: pd.DataFrame) -> list[str]:
    issues = []

    missing_cols = [c for c in EXPERIMENTAL_COLUMNS if c not in df.columns]
    if missing_cols:
        issues.append(f"Kolom wajib belum ada: {missing_cols}")

    if len(df) == 0:
        issues.append("Dataset kosong (0 baris).")

    if "pH_reference" in df.columns and len(df) > 0:
        out_of_range = df[(df["pH_reference"] < 0) | (df["pH_reference"] > 14)]
        if len(out_of_range) > 0:
            issues.append(f"{len(out_of_range)} baris punya pH_reference di luar rentang 0-14.")

    key_cols = [c for c in ["sample_id", "R_mean", "G_mean", "B_mean", "pH_reference"] if c in df.columns]
    if key_cols and len(df) > 0:
        missing_values = df[key_cols].isna().sum().sum()
        if missing_values > 0:
            issues.append(f"Ada {missing_values} nilai kosong di kolom kunci {key_cols}.")

    if issues:
        raise ExperimentalDataError("; ".join(issues))

    return issues
