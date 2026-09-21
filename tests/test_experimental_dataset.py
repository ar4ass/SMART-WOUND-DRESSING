import os
import shutil
import tempfile

import pandas as pd
import pytest

from src.experimental_dataset import (
    EXPERIMENTAL_COLUMNS,
    ExperimentalDataError,
    generate_template,
    load_experimental_dataset,
    validate_experimental_dataframe,
)


@pytest.fixture()
def temp_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_generate_template_has_correct_columns(temp_dir):
    path = os.path.join(temp_dir, "template.csv")
    generate_template(path)

    df = pd.read_csv(path)
    assert list(df.columns) == EXPERIMENTAL_COLUMNS
    assert len(df) == 0


def _valid_row(**overrides):
    row = {
        "sample_id": "S001", "date": "2026-09-21", "material_id": "M001",
        "pH_reference": 6.5, "image_path": "img.jpg",
        "R_mean": 150.0, "G_mean": 100.0, "B_mean": 90.0,
        "H_mean": 10.0, "S_mean": 100.0, "V_mean": 150.0,
        "lighting_condition": "indoor", "smartphone_model": "Phone X",
        "distance_cm": 15, "angle": 90, "replicate": 1,
    }
    row.update(overrides)
    return row


def test_valid_dataframe_passes_validation():
    df = pd.DataFrame([_valid_row(), _valid_row(sample_id="S002")])
    issues = validate_experimental_dataframe(df)
    assert issues == []


def test_missing_columns_raise_error():
    df = pd.DataFrame([{"sample_id": "S001"}])
    with pytest.raises(ExperimentalDataError):
        validate_experimental_dataframe(df)


def test_ph_out_of_range_raises_error():
    df = pd.DataFrame([_valid_row(pH_reference=20)])
    with pytest.raises(ExperimentalDataError):
        validate_experimental_dataframe(df)


def test_empty_dataframe_raises_error():
    df = pd.DataFrame(columns=EXPERIMENTAL_COLUMNS)
    with pytest.raises(ExperimentalDataError):
        validate_experimental_dataframe(df)


def test_load_experimental_dataset_roundtrip(temp_dir):
    path = os.path.join(temp_dir, "data.csv")
    pd.DataFrame([_valid_row(), _valid_row(sample_id="S002")]).to_csv(path, index=False)

    df = load_experimental_dataset(path)
    assert len(df) == 2


def test_load_invalid_dataset_raises_error(temp_dir):
    path = os.path.join(temp_dir, "bad.csv")
    pd.DataFrame([{"sample_id": "S001"}]).to_csv(path, index=False)

    with pytest.raises(ExperimentalDataError):
        load_experimental_dataset(path)
