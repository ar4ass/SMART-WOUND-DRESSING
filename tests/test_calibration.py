import numpy as np
import pytest

from src.calibration import CalibrationResult, train_and_evaluate_models
from src.simulation_dataset import generate_simulation_dataset


def test_train_and_evaluate_models_returns_four_models():
    df = generate_simulation_dataset(n_samples=60, random_seed=1)
    X = df[["R_mean", "G_mean", "B_mean", "H_mean", "S_mean", "V_mean"]].values
    y = df["target_value"].values

    result = train_and_evaluate_models(X, y, dataset_type="simulation")

    assert isinstance(result, CalibrationResult)
    assert set(result.models.keys()) == {
        "linear_regression", "polynomial_regression", "random_forest", "svr"
    }
    assert set(result.metrics.keys()) == set(result.models.keys())
    assert result.best_model_name in result.models


def test_result_flags_simulation_dataset_correctly():
    df = generate_simulation_dataset(n_samples=40, random_seed=2)
    X = df[["R_mean", "G_mean", "B_mean", "H_mean", "S_mean", "V_mean"]].values
    y = df["target_value"].values

    result = train_and_evaluate_models(X, y, dataset_type="simulation")

    assert result.is_simulation_only is True


def test_invalid_dataset_type_raises_error():
    X = np.random.rand(20, 6)
    y = np.random.rand(20)
    with pytest.raises(ValueError):
        train_and_evaluate_models(X, y, dataset_type="not_a_real_type")


def test_trained_model_can_predict():
    df = generate_simulation_dataset(n_samples=60, random_seed=5)
    X = df[["R_mean", "G_mean", "B_mean", "H_mean", "S_mean", "V_mean"]].values
    y = df["target_value"].values

    result = train_and_evaluate_models(X, y, dataset_type="simulation")
    best_model = result.models[result.best_model_name]

    prediction = best_model.predict(X[:1])
    assert prediction.shape == (1,)
