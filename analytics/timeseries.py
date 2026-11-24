"""Read-only time-series preparation and temporal aggregation helpers."""
from __future__ import annotations

import pandas as pd
import numpy as np
import warnings
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller


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


def park_wide_series(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate processed lot records into observed half-hour park-wide values."""
    required = {"slot", "Occupancy", "Capacity", "QueueLength", "occ_rate"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Processed feature data is missing columns: {sorted(missing)}")
    result = data.groupby("slot").agg(
        occupancy=("Occupancy", "sum"), capacity=("Capacity", "sum"),
        queue_length=("QueueLength", "mean"),
    ).sort_index()
    result["occupancy_rate"] = result.occupancy / result.capacity.replace(0, np.nan)
    result.index = pd.to_datetime(result.index)
    result.index.name = "timestamp"
    return result


def daily_park_series(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate park-wide values by dates represented in the data; never fill dates."""
    series = park_wide_series(data)
    return series.groupby(series.index.normalize()).agg(
        occupancy=("occupancy", "mean"), occupancy_rate=("occupancy_rate", "mean"),
        queue_length=("queue_length", "mean"), slots=("occupancy_rate", "count"),
    ).rename_axis("date").sort_index()


def longest_contiguous_daily_segment(daily: pd.DataFrame, minimum_days: int = 14) -> pd.DataFrame:
    """Get the longest uninterrupted calendar run without interpolating missing dates."""
    if daily.empty:
        return daily
    ordered = daily.sort_index()
    dates = pd.DatetimeIndex(ordered.index).normalize()
    gap_days = np.diff(dates.values).astype("timedelta64[D]").astype(int)
    segments = np.split(np.arange(len(ordered)), np.flatnonzero(gap_days != 1) + 1)
    longest = max(segments, key=len)
    return ordered.iloc[longest] if len(longest) >= minimum_days else ordered.iloc[0:0]


def hourly_seasonality(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed occupancy by hour of day."""
    series = park_wide_series(data)
    frame = series.assign(hour=series.index.hour)
    return frame.groupby("hour").occupancy_rate.agg(["count", "mean", "std"]).reset_index()


def weekday_seasonality(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize park-wide occupancy by day of week using observed slots."""
    series = park_wide_series(data)
    frame = series.assign(day_of_week=series.index.dayofweek,
                          day_name=series.index.day_name())
    return (frame.groupby(["day_of_week", "day_name"]).occupancy_rate
            .agg(["count", "mean", "std"]).reset_index().sort_values("day_of_week"))


def adf_stationarity_test(series: pd.Series) -> dict:
    """Run ADF on a supplied observed regular series."""
    values = pd.Series(series).dropna().astype(float)
    if len(values) < 10:
        return {"available": False, "reason": "ADF requires at least 10 observed values."}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        result = adfuller(values, autolag="AIC")
    return {"available": True, "statistic": float(result[0]), "p_value": float(result[1]),
            "used_lags": int(result[2]), "observations": int(result[3]),
            "critical_values": {k: float(v) for k, v in result[4].items()},
            "is_stationary_at_5pct": bool(result[1] < .05)}


def stl_daily_decomposition(daily: pd.DataFrame, period: int = 7) -> dict:
    """STL-decompose the longest uninterrupted daily occupancy-rate segment."""
    segment = longest_contiguous_daily_segment(daily, minimum_days=period * 2)
    if len(segment) < period * 2:
        return {"available": False,
                "reason": f"Need at least {period * 2} uninterrupted observed days for period-{period} STL."}
    result = STL(segment["occupancy_rate"].astype(float), period=period, robust=True).fit()
    components = pd.DataFrame({"observed": segment.occupancy_rate,
                               "trend": result.trend, "seasonal": result.seasonal,
                               "residual": result.resid}, index=segment.index)
    return {"available": True, "period": period, "observations": len(segment),
            "start": segment.index.min(), "end": segment.index.max(), "components": components}
