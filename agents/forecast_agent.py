"""Forecast agent: predicts each lot's occupancy rate for the next 6 slots (3 hours).
One XGBoost model per step ahead."""
import joblib
import numpy as np
from xgboost import XGBRegressor

from agents.data_agent import FEATURES
from common import FORECAST_MODEL_PATH

PARAMS = dict(learning_rate=0.05, max_depth=5, subsample=0.8, colsample_bytree=0.8,
              random_state=42, n_jobs=-1)


def make_target(df, h):
    """Occupancy rate h slots later for the same lot."""
    return df.groupby("SystemCodeNumber")["occ_rate"].shift(-h)


def fit_xgb(df, h, train_mask, val_mask=None, n_estimators=None):
    """With val_mask: early stopping picks the number of trees. Without: use n_estimators."""
    y = make_target(df, h)
    tr = train_mask & y.notna()
    if val_mask is None:
        model = XGBRegressor(n_estimators=n_estimators, **PARAMS)
        model.fit(df.loc[tr, FEATURES], y[tr])
    else:
        va = val_mask & y.notna()
        model = XGBRegressor(n_estimators=1000, early_stopping_rounds=50, **PARAMS)
        model.fit(df.loc[tr, FEATURES], y[tr],
                  eval_set=[(df.loc[va, FEATURES], y[va])], verbose=False)
    return model


class ForecastAgent:
    def __init__(self, horizon=6):
        saved = joblib.load(FORECAST_MODEL_PATH)
        self.models, self.n_trees, self.test_mae = saved["models"], saved["n_trees"], saved["test_mae"]
        self.horizon = horizon

    def predict(self, batch):
        """{lot: [6 predicted occupancy rates, next slot first]}"""
        preds = [np.clip(self.models[h].predict(batch[FEATURES]), 0, 1)
                 for h in range(1, self.horizon + 1)]
        return {lot: [float(p[i]) for p in preds]
                for i, lot in enumerate(batch["SystemCodeNumber"])}

    def retrain(self, rows):
        """Refit on everything seen so far (called when the monitor sees drift)."""
        mask = rows["occ_rate"].notna()
        for h in range(1, self.horizon + 1):
            self.models[h] = fit_xgb(rows, h, mask, n_estimators=self.n_trees[h])
