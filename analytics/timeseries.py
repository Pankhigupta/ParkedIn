"""Read-only time-series preparation and temporal aggregation helpers."""
from __future__ import annotations

import pandas as pd


def occupancy_series(data: pd.DataFrame, lot_id: str | None = None) -> pd.DataFrame:
    """Return timestamped occupancy records from raw or engineered project data."""
    frame = data.copy()
    if "slot" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["slot"])
    elif {"LastUpdatedDate", "LastUpdatedTime"}.issubset(frame.columns):
        frame["timestamp"] = pd.to_datetime(
            frame["LastUpdatedDate"].astype(str) + " " + frame["LastUpdatedTime"].astype(str),
            format="%d-%m-%Y %H:%M:%S", errors="coerce")
    else:
        raise ValueError("Data must contain 'slot' or raw LastUpdatedDate/LastUpdatedTime columns")

    lot_column = "SystemCodeNumber" if "SystemCodeNumber" in frame.columns else None
    if lot_id is not None:
        if lot_column is None:
            raise ValueError("Lot filtering requires the SystemCodeNumber column")
        frame = frame[frame[lot_column].astype(str) == str(lot_id)]

    if "occ_rate" not in frame.columns:
        if not {"Occupancy", "Capacity"}.issubset(frame.columns):
            raise ValueError("Data must contain occ_rate or Occupancy and Capacity")
        frame["occ_rate"] = frame["Occupancy"] / frame["Capacity"].replace(0, pd.NA)
    keep = ["timestamp", "occ_rate"] + ([lot_column] if lot_column else [])
    return frame[keep].dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def temporal_occupancy_summary(data: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Aggregate mean occupancy by hour-of-day and day-of-week."""
    series = occupancy_series(data)
    series["hour"] = series.timestamp.dt.hour
    series["day_of_week"] = series.timestamp.dt.day_name()
    return {
        "by_hour": series.groupby("hour").occ_rate.agg(["count", "mean", "std"]).reset_index(),
        "by_day_of_week": series.groupby("day_of_week", sort=False).occ_rate
        .agg(["count", "mean", "std"]).reset_index(),
    }


def regularity_summary(data: pd.DataFrame) -> dict:
    """Describe observed timestamp spacing without filling or resampling readings."""
    series = occupancy_series(data)
    deltas = series.timestamp.sort_values().drop_duplicates().diff().dropna()
    return {"records": int(len(series)), "first_timestamp": series.timestamp.min(),
            "last_timestamp": series.timestamp.max(),
            "median_interval": deltas.median() if not deltas.empty else pd.NaT,
            "unique_intervals": int(deltas.nunique())}
