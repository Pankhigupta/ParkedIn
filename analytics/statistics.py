"""Dataset and model inventory plus reusable descriptive-statistics helpers.

This module only reads existing files. It does not rebuild features, train models,
or change the forecasting and pricing pipelines.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing

import pandas as pd

from common import (DATA_RAW, FEATURES_PATH, FORECAST_MODEL_PATH, METRICS_PATH,
                    DB_PATH, MODELS_DIR, PATTERN_MODEL_PATH)

PRICING_RESULTS_PATH = MODELS_DIR / "pricing_results.csv"


def dataset_catalog() -> list[dict]:
    """Describe source and engineered datasets currently used by the project."""
    sources = [
        ("raw", "Raw parking readings", DATA_RAW,
         "Original parking dataset used as input to feature preparation."),
        ("processed", "Engineered feature dataset", FEATURES_PATH,
         "Cleaned half-hour lot records with occupancy features and train/validation/test split."),
        ("simulation_db", "Runtime simulation database", DB_PATH,
         "SQLite store for replayed readings, forecasts, prices, metrics, alerts, and controls."),
    ]
    return [{"key": key, "name": name, "path": str(path), "exists": path.is_file(),
             "description": description,
             "size_bytes": path.stat().st_size if path.is_file() else None}
            for key, name, path, description in sources]


def load_dataset(source: str = "processed") -> pd.DataFrame:
    """Load an existing raw CSV or processed feature pickle without generating data."""
    if source == "raw":
        path, reader = DATA_RAW, pd.read_csv
    elif source == "processed":
        path, reader = FEATURES_PATH, pd.read_pickle
    else:
        raise ValueError("source must be 'raw' or 'processed'")
    if not path.is_file():
        raise FileNotFoundError(f"{source} dataset not found at {path}")
    return reader(path)


def load_simulation_table(table: str) -> pd.DataFrame:
    """Read one known runtime table from SQLite without creating or changing the DB."""
    allowed_tables = {"lots", "readings", "forecasts", "prices", "metrics", "alerts", "control"}
    if table not in allowed_tables:
        raise ValueError(f"table must be one of: {', '.join(sorted(allowed_tables))}")
    if not DB_PATH.is_file():
        raise FileNotFoundError(f"Runtime simulation database not found at {DB_PATH}")
    uri = f"{DB_PATH.resolve().as_uri()}?mode=ro"
    with closing(sqlite3.connect(uri, uri=True)) as connection:
        return pd.read_sql_query(f'SELECT * FROM "{table}"', connection)


def model_catalog() -> list[dict]:
    """Inventory persisted trained models and evaluation artifacts."""
    records = [
        ("forecast", "XGBoost occupancy forecasters", FORECAST_MODEL_PATH, "trained_model",
         "Serialized bundle with one forecast model per horizon, tree counts, and test MAE."),
        ("pattern", "K-Means demand classifier", PATTERN_MODEL_PATH, "trained_model",
         "Serialized K-Means model, StandardScaler, and demand-level names."),
        ("evaluation_metrics", "Forecast and clustering metrics", METRICS_PATH, "evaluation_artifact",
         "MAE comparisons, feature importance, K-Means inertia, silhouette, and hourly levels."),
        ("pricing_results", "Pricing strategy comparison", PRICING_RESULTS_PATH, "evaluation_artifact",
         "Offline pricing strategy results across assumed price sensitivities."),
    ]
    return [{"key": key, "name": name, "path": str(path), "exists": path.is_file(),
             "artifact_type": kind, "description": description,
             "size_bytes": path.stat().st_size if path.is_file() else None}
            for key, name, path, kind, description in records]


def dataset_statistics(data: pd.DataFrame) -> dict:
    """Return compact structural and descriptive statistics for a dataframe."""
    numeric = data.select_dtypes(include="number")
    return {
        "rows": int(data.shape[0]),
        "columns": int(data.shape[1]),
        "column_names": list(data.columns),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing_cells": int(data.isna().sum().sum()),
        "missing_percent": round(float(data.isna().to_numpy().mean() * 100), 3)
        if data.size else 0.0,
        "numeric_summary": numeric.describe().round(4).to_dict(),
    }


def load_evaluation_metrics() -> dict:
    """Read train.py's metrics artifact, if training has already produced it."""
    if not METRICS_PATH.is_file():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))
