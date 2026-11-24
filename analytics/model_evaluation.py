"""Read existing model-comparison artifacts without loading or retraining models."""
from __future__ import annotations

import pandas as pd

from common import MODELS_DIR
from .statistics import load_evaluation_metrics


def forecast_comparison() -> pd.DataFrame:
    """Return forecast MAE by horizon and model from the saved training metrics."""
    metrics = load_evaluation_metrics().get("forecast_mae", {})
    if not metrics:
        return pd.DataFrame()
    frame = pd.DataFrame(metrics)
    frame.index.name = "slots_ahead"
    frame.index = pd.to_numeric(frame.index, errors="coerce")
    return frame.sort_index()


def feature_importance() -> pd.Series:
    """Return saved one-step XGBoost feature importances, if available."""
    values = load_evaluation_metrics().get("feature_importance", {})
    return pd.Series(values, dtype="float64", name="importance").sort_values(ascending=False)


def clustering_evaluation() -> dict:
    """Return saved K-Means validation summaries."""
    metrics = load_evaluation_metrics()
    return {"elbow_inertia": metrics.get("kmeans_elbow_inertia", {}),
            "silhouette_k4": metrics.get("kmeans_silhouette_k4"),
            "levels_by_hour": metrics.get("levels_by_hour", {})}


def pricing_comparison() -> pd.DataFrame:
    """Read evaluate_pricing.py's output table, if it exists."""
    path = MODELS_DIR / "pricing_results.csv"
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()
