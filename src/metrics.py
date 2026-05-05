from __future__ import annotations
from typing import Any
from sklearn.metrics import accuracy_score, f1_score


def compute_metrics(y_true: Any, y_pred: Any) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted")
    }
