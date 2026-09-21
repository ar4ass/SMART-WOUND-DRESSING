from __future__ import annotations

import csv
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

HISTORY_COLUMNS = [
    "timestamp",
    "image_name",
    "mean_r",
    "mean_g",
    "mean_b",
    "mean_h",
    "mean_s",
    "mean_v",
    "predicted_value",
    "model_name",
    "dataset_type",
]


@dataclass
class HistoryEntry:
    timestamp: str
    image_name: str
    mean_r: float
    mean_g: float
    mean_b: float
    mean_h: float
    mean_s: float
    mean_v: float
    predicted_value: Optional[float]
    model_name: Optional[str]
    dataset_type: str


def _ensure_file(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
            writer.writeheader()


def add_entry(
    image_name: str,
    features: dict,
    predicted_value: Optional[float] = None,
    model_name: Optional[str] = None,
    dataset_type: str = "no_model",
    path: str = "data/processed/history.csv",
) -> HistoryEntry:
    entry = HistoryEntry(
        timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        image_name=image_name,
        mean_r=round(features.get("mean_r", 0.0), 2),
        mean_g=round(features.get("mean_g", 0.0), 2),
        mean_b=round(features.get("mean_b", 0.0), 2),
        mean_h=round(features.get("mean_h", 0.0), 2),
        mean_s=round(features.get("mean_s", 0.0), 2),
        mean_v=round(features.get("mean_v", 0.0), 2),
        predicted_value=round(predicted_value, 3) if predicted_value is not None else None,
        model_name=model_name,
        dataset_type=dataset_type,
    )

    _ensure_file(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_COLUMNS)
        writer.writerow(asdict(entry))

    return entry


def read_history(path: str = "data/processed/history.csv") -> list[dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def clear_history(path: str = "data/processed/history.csv") -> None:
    if os.path.exists(path):
        os.remove(path)
