import pandas as pd

from src.simulation_dataset import DATASET_TYPE_LABEL, generate_simulation_dataset


def test_dataset_has_expected_columns_and_length():
    df = generate_simulation_dataset(n_samples=20, random_seed=1)

    assert len(df) == 20
    expected_cols = {
        "image_id", "data_type", "target_value",
        "R_mean", "G_mean", "B_mean", "H_mean", "S_mean", "V_mean",
    }
    assert expected_cols.issubset(df.columns)


def test_every_row_is_labeled_simulation():
    df = generate_simulation_dataset(n_samples=10, random_seed=1)
    assert (df["data_type"] == DATASET_TYPE_LABEL).all()


def test_same_seed_gives_reproducible_data():
    df1 = generate_simulation_dataset(n_samples=15, random_seed=7)
    df2 = generate_simulation_dataset(n_samples=15, random_seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_different_seed_gives_different_data():
    df1 = generate_simulation_dataset(n_samples=15, random_seed=1)
    df2 = generate_simulation_dataset(n_samples=15, random_seed=2)
    assert not df1["target_value"].equals(df2["target_value"])


def test_rgb_values_within_valid_range():
    df = generate_simulation_dataset(n_samples=50, random_seed=3)
    for col in ["R_mean", "G_mean", "B_mean"]:
        assert df[col].between(0, 255).all()
