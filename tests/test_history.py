import os
import shutil
import tempfile

import pytest

from src.history import add_entry, clear_history, read_history


@pytest.fixture()
def temp_history_path():
    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, "history.csv")
    yield path
    shutil.rmtree(tmp_dir, ignore_errors=True)


def test_add_entry_creates_file_with_header(temp_history_path):
    features = {"mean_r": 100.0, "mean_g": 90.0, "mean_b": 80.0, "mean_h": 10.0, "mean_s": 50.0, "mean_v": 100.0}

    add_entry("test.jpg", features, predicted_value=None, dataset_type="no_model", path=temp_history_path)

    assert os.path.exists(temp_history_path)
    rows = read_history(temp_history_path)
    assert len(rows) == 1
    assert rows[0]["image_name"] == "test.jpg"


def test_multiple_entries_accumulate(temp_history_path):
    features = {"mean_r": 1, "mean_g": 2, "mean_b": 3, "mean_h": 4, "mean_s": 5, "mean_v": 6}

    for i in range(3):
        add_entry(f"img_{i}.jpg", features, path=temp_history_path)

    rows = read_history(temp_history_path)
    assert len(rows) == 3
    assert [r["image_name"] for r in rows] == ["img_0.jpg", "img_1.jpg", "img_2.jpg"]


def test_read_history_empty_when_no_file(temp_history_path):
    assert read_history(temp_history_path) == []


def test_clear_history_removes_file(temp_history_path):
    features = {"mean_r": 1, "mean_g": 2, "mean_b": 3, "mean_h": 4, "mean_s": 5, "mean_v": 6}
    add_entry("img.jpg", features, path=temp_history_path)
    assert os.path.exists(temp_history_path)

    clear_history(temp_history_path)
    assert not os.path.exists(temp_history_path)
    assert read_history(temp_history_path) == []


def test_entry_with_prediction_stores_model_info(temp_history_path):
    features = {"mean_r": 1, "mean_g": 2, "mean_b": 3, "mean_h": 4, "mean_s": 5, "mean_v": 6}
    add_entry(
        "img.jpg", features, predicted_value=6.5, model_name="random_forest",
        dataset_type="simulation", path=temp_history_path,
    )
    rows = read_history(temp_history_path)
    assert rows[0]["predicted_value"] == "6.5"
    assert rows[0]["dataset_type"] == "simulation"
