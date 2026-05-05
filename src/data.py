from __future__ import annotations
from typing import Any
import pandas as pd
from sklearn.model_selection import train_test_split
from config import DATA_DIR


def load_dataset_split() -> tuple[Any, Any, Any, Any]:
    df = pd.read_csv(DATA_DIR / "processed_dataset.csv")
    X = df.drop(columns=["good_rating"])
    y = df["good_rating"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

