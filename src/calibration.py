from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVR

FEATURE_COLUMNS = ["R_mean", "G_mean", "B_mean", "H_mean", "S_mean", "V_mean"]


def _build_models() -> dict:
    return {
        "linear_regression": LinearRegression(),
        "polynomial_regression": make_pipeline(
            PolynomialFeatures(degree=2), LinearRegression()
        ),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "svr": make_pipeline(StandardScaler(), SVR(kernel="rbf")),
    }


@dataclass
class ModelMetrics:
    mae: float
    rmse: float
    r2: float


@dataclass
class CalibrationResult:
    dataset_type: str
    models: Dict[str, object] = field(default_factory=dict)
    metrics: Dict[str, ModelMetrics] = field(default_factory=dict)
    best_model_name: str = ""

    @property
    def is_simulation_only(self) -> bool:
        return self.dataset_type == "simulation"


def train_and_evaluate_models(
    X: np.ndarray, y: np.ndarray, dataset_type: str, test_size: float = 0.2, random_state: int = 42
) -> CalibrationResult:
    if dataset_type not in ("simulation", "experimental"):
        raise ValueError('dataset_type harus "simulation" atau "experimental".')

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    result = CalibrationResult(dataset_type=dataset_type)

    best_r2 = -np.inf
    for name, model in _build_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = ModelMetrics(
            mae=float(mean_absolute_error(y_test, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
            r2=float(r2_score(y_test, y_pred)),
        )

        result.models[name] = model
        result.metrics[name] = metrics

        if metrics.r2 > best_r2:
            best_r2 = metrics.r2
            result.best_model_name = name

    return result


def metrics_to_table(result: CalibrationResult) -> list[dict]:
    rows = []
    for name, m in result.metrics.items():
        rows.append(
            {
                "model": name,
                "MAE": round(m.mae, 4),
                "RMSE": round(m.rmse, 4),
                "R2": round(m.r2, 4),
                "is_best": name == result.best_model_name,
            }
        )
    return rows
