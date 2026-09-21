from __future__ import annotations

import numpy as np
import pandas as pd

DATASET_TYPE_LABEL = "simulation"


def generate_simulation_dataset(n_samples: int = 100, random_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)

    target_value = rng.uniform(4.0, 9.0, size=n_samples)

    noise_scale = 15.0
    r_mean = np.clip(220 - (target_value - 4.0) * 20 + rng.normal(0, noise_scale, n_samples), 0, 255)
    g_mean = np.clip(150 + rng.normal(0, noise_scale, n_samples), 0, 255)
    b_mean = np.clip(60 + (target_value - 4.0) * 18 + rng.normal(0, noise_scale, n_samples), 0, 255)

    import cv2

    rgb_pixels = np.stack([r_mean, g_mean, b_mean], axis=1).astype(np.uint8).reshape(1, -1, 3)
    hsv_pixels = cv2.cvtColor(rgb_pixels, cv2.COLOR_RGB2HSV).reshape(-1, 3)

    df = pd.DataFrame(
        {
            "image_id": [f"sample_{i:03d}" for i in range(1, n_samples + 1)],
            "data_type": DATASET_TYPE_LABEL,
            "target_value": np.round(target_value, 2),
            "R_mean": np.round(r_mean, 1),
            "G_mean": np.round(g_mean, 1),
            "B_mean": np.round(b_mean, 1),
            "H_mean": hsv_pixels[:, 0],
            "S_mean": hsv_pixels[:, 1],
            "V_mean": hsv_pixels[:, 2],
        }
    )
    return df


def save_simulation_dataset(path: str, n_samples: int = 100, random_seed: int = 42) -> pd.DataFrame:
    df = generate_simulation_dataset(n_samples=n_samples, random_seed=random_seed)
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    output_path = "data/sample/simulation_dataset.csv"
    df = save_simulation_dataset(output_path, n_samples=150)
    print(f"Dataset simulasi disimpan ke {output_path} ({len(df)} baris).")
    print(df.head())
