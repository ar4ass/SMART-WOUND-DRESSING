from __future__ import annotations

import os
from dataclasses import dataclass

import joblib
import numpy as np

from src.calibration import FEATURE_COLUMNS

FEATURE_KEY_MAP = {
    "R_mean": "mean_r",
    "G_mean": "mean_g",
    "B_mean": "mean_b",
    "H_mean": "mean_h",
    "S_mean": "mean_s",
    "V_mean": "mean_v",
}


class ModelNotAvailableError(Exception):
    pass


@dataclass
class PredictionResult:
    value: float
    model_name: str
    is_simulation_only: bool


def save_model(model, model_name: str, dataset_type: str, models_dir: str = "models") -> str:
    os.makedirs(models_dir, exist_ok=True)
    filename = f"{model_name}__{dataset_type}.joblib"
    path = os.path.join(models_dir, filename)
    joblib.dump(model, path)
    return path


def load_model(model_name: str, dataset_type: str, models_dir: str = "models"):
    filename = f"{model_name}__{dataset_type}.joblib"
    path = os.path.join(models_dir, filename)
    if not os.path.exists(path):
        raise ModelNotAvailableError(
            f"Model '{model_name}' ({dataset_type}) belum tersedia di {path}. "
            "Latih model dulu lewat STEP 10 / halaman training."
        )
    return joblib.load(path)


def predict_from_features(features: dict, model, model_name: str, dataset_type: str) -> PredictionResult:
    missing = [
        col for col in FEATURE_COLUMNS
        if col not in features and FEATURE_KEY_MAP.get(col) not in features
    ]
    if missing:
        raise ValueError(f"Fitur berikut belum ada di 'features': {missing}")

    def _get(col: str) -> float:
        if col in features:
            return features[col]
        return features[FEATURE_KEY_MAP[col]]

    X = np.array([[_get(col) for col in FEATURE_COLUMNS]])
    value = float(model.predict(X)[0])

    return PredictionResult(
        value=value,
        model_name=model_name,
        is_simulation_only=(dataset_type == "simulation"),
    )
