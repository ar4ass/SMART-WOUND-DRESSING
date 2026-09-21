import shutil
import tempfile

import pytest

from src.calibration import train_and_evaluate_models
from src.prediction import (
    ModelNotAvailableError,
    load_model,
    predict_from_features,
    save_model,
)
from src.simulation_dataset import generate_simulation_dataset


@pytest.fixture()
def temp_models_dir():
    path = tempfile.mkdtemp()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def trained_result():
    df = generate_simulation_dataset(n_samples=60, random_seed=9)
    X = df[["R_mean", "G_mean", "B_mean", "H_mean", "S_mean", "V_mean"]].values
    y = df["target_value"].values
    return train_and_evaluate_models(X, y, dataset_type="simulation")


def test_save_and_load_model_roundtrip(trained_result, temp_models_dir):
    model = trained_result.models[trained_result.best_model_name]

    path = save_model(model, trained_result.best_model_name, "simulation", models_dir=temp_models_dir)
    assert path.endswith(".joblib")

    loaded = load_model(trained_result.best_model_name, "simulation", models_dir=temp_models_dir)
    prediction = loaded.predict([[150, 120, 90, 10, 100, 150]])
    assert len(prediction) == 1


def test_load_missing_model_raises_error(temp_models_dir):
    with pytest.raises(ModelNotAvailableError):
        load_model("random_forest", "simulation", models_dir=temp_models_dir)


def test_predict_from_features_marks_simulation_only(trained_result):
    model = trained_result.models[trained_result.best_model_name]
    features = {
        "mean_r": 150.0, "mean_g": 120.0, "mean_b": 90.0,
        "mean_h": 10.0, "mean_s": 100.0, "mean_v": 150.0,
    }

    result = predict_from_features(features, model, trained_result.best_model_name, "simulation")

    assert result.is_simulation_only is True
    assert isinstance(result.value, float)


def test_predict_from_features_missing_key_raises_error(trained_result):
    model = trained_result.models[trained_result.best_model_name]
    incomplete_features = {"mean_r": 150.0}

    with pytest.raises(ValueError):
        predict_from_features(incomplete_features, model, trained_result.best_model_name, "simulation")
