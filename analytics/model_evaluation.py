"""Read existing model-comparison artifacts without loading or retraining models."""
from __future__ import annotations

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from agents.data_agent import FEATURES
from agents.forecast_agent import make_target
from common import FORECAST_MODEL_PATH, MODELS_DIR
from .statistics import load_dataset
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


def implemented_model_catalog() -> list[dict]:
    """Document implemented methods and whether their fitted models are persisted."""
    return [
        {"model": "XGBoost Regressor", "family": "Forecasting regression / boosted-tree ensemble",
         "artifact_status": "Persisted: six horizon-specific models", "evaluation": "Full saved MAE comparison; other metrics can be recomputed from frozen models."},
        {"model": "Random Forest Regressor", "family": "Forecasting regression / bagged-tree ensemble",
         "artifact_status": "Not persisted", "evaluation": "Saved test MAE by horizon only."},
        {"model": "ARIMA(2,0,1)", "family": "Statistical time-series forecasting",
         "artifact_status": "Not persisted", "evaluation": "Saved test MAE by horizon only."},
        {"model": "Naive persistence", "family": "Forecasting baseline", "artifact_status": "No fitted artifact required",
         "evaluation": "Saved test MAE by horizon; full metrics can be recomputed from the held-out series."},
        {"model": "K-Means (k=4)", "family": "Unsupervised clustering", "artifact_status": "Persisted with StandardScaler",
         "evaluation": "Saved silhouette score and cluster diagnostics; not comparable to forecast error."},
    ]


def _forecast_metric_record(y_true, y_pred, model: str, horizon: int) -> dict:
    y = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    valid = np.isfinite(y) & np.isfinite(pred)
    y, pred = y[valid], pred[valid]
    nonzero = np.abs(y) > np.finfo(float).eps
    mape = float(np.mean(np.abs((y[nonzero] - pred[nonzero]) / y[nonzero])) * 100) if nonzero.any() else np.nan
    denominator = np.abs(y) + np.abs(pred)
    smape_terms = np.divide(2 * np.abs(y-pred), denominator,
                            out=np.zeros_like(denominator), where=denominator > 0)
    return {"model": model, "horizon": int(horizon), "n": int(len(y)),
            "MAE": float(mean_absolute_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R²": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
            "MAPE (%)": mape, "SMAPE (%)": float(smape_terms.mean() * 100),
            "MAPE denominator rows": int(nonzero.sum()),
            "source": "Recomputed from saved model / held-out target"}


def frozen_forecast_metrics() -> pd.DataFrame:
    """Compute full metrics from persisted XGBoost and naive predictions without fitting."""
    if not FORECAST_MODEL_PATH.is_file():
        return pd.DataFrame()
    data = load_dataset("processed")
    saved = joblib.load(FORECAST_MODEL_PATH)
    test = data["split"].eq("test")
    complete = data[FEATURES].notna().all(axis=1)
    records = []
    for horizon, model in sorted(saved["models"].items()):
        target = make_target(data, int(horizon))
        mask = test & target.notna() & complete
        y = target.loc[mask].to_numpy(dtype=float)
        x = data.loc[mask, FEATURES]
        xgb_prediction = np.clip(model.predict(x), 0, 1)
        naive_prediction = data.loc[mask, "occ_rate"].to_numpy(dtype=float)
        records.append(_forecast_metric_record(y, xgb_prediction, "XGBoost (saved)", int(horizon)))
        records.append(_forecast_metric_record(y, naive_prediction, "Naive", int(horizon)))
    return pd.DataFrame(records)
