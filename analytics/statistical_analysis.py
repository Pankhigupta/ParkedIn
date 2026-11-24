"""Statistical analyses for the real ParkedIn source dataset.

The functions below derive a working analysis frame from the raw CSV. They do not
write files, train models, alter the forecasting pipeline, or generate synthetic rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .statistics import load_dataset


def load_analysis_data() -> pd.DataFrame:
    """Load the project's actual CSV and add analysis-only date/rate columns."""
    frame = load_dataset("raw")
    frame["timestamp"] = pd.to_datetime(
        frame["LastUpdatedDate"].astype(str) + " " + frame["LastUpdatedTime"].astype(str),
        format="%d-%m-%Y %H:%M:%S", errors="coerce")
    frame["date"] = frame["timestamp"].dt.normalize()
    frame["OccupancyRate"] = frame["Occupancy"] / frame["Capacity"].replace(0, np.nan)
    frame["OccupancyRate"] = frame["OccupancyRate"].clip(0, 1)
    frame["hour"] = frame["timestamp"].dt.hour
    frame["day_of_week"] = frame["timestamp"].dt.dayofweek
    frame["day_name"] = frame["timestamp"].dt.day_name()
    frame["is_weekend"] = frame["day_of_week"] >= 5
    frame["traffic_code"] = frame["TrafficConditionNearby"].map(
        {"low": 0, "average": 1, "high": 2})
    return frame.dropna(subset=["timestamp", "OccupancyRate"]).reset_index(drop=True)


def _decision(p_value: float, alpha: float = .05) -> str:
    return "Reject the null hypothesis at α = 0.05." if p_value < alpha else \
        "Fail to reject the null hypothesis at α = 0.05; evidence is insufficient to establish a difference or relationship."


def traffic_occupancy_anova(frame: pd.DataFrame) -> dict:
    groups = {str(label): values["OccupancyRate"].dropna().to_numpy()
              for label, values in frame.groupby("TrafficConditionNearby", dropna=True)}
    groups = {name: values for name, values in groups.items() if len(values) >= 2}
    if len(groups) < 2:
        return {"available": False, "reason": "Fewer than two traffic categories have enough observations."}
    result = stats.f_oneway(*groups.values())
    grand = float(frame["OccupancyRate"].mean())
    ss_between = sum(len(values) * (float(values.mean()) - grand) ** 2 for values in groups.values())
    ss_total = float(((frame["OccupancyRate"] - grand) ** 2).sum())
    return {"available": True, "statistic": float(result.statistic), "p_value": float(result.pvalue),
            "eta_squared": ss_between / ss_total if ss_total else 0.0,
            "groups": {name: {"n": int(len(values)), "mean": float(values.mean()),
                              "median": float(np.median(values)), "std": float(values.std(ddof=1))}
                       for name, values in groups.items()},
            "decision": _decision(float(result.pvalue))}


def special_day_ttest(frame: pd.DataFrame) -> dict:
    # Daily means provide one observation per date rather than treating every row as independent.
    daily = frame.groupby("date").agg(
        occupancy=("OccupancyRate", "mean"), special_day=("IsSpecialDay", "max")).dropna()
    regular = daily.loc[daily.special_day == 0, "occupancy"].to_numpy()
    special = daily.loc[daily.special_day == 1, "occupancy"].to_numpy()
    if len(regular) < 2 or len(special) < 2:
        return {"available": False, "reason": "At least two regular and two special-day dates are needed."}
    result = stats.ttest_ind(special, regular, equal_var=False)
    difference = float(special.mean() - regular.mean())
    v1, v0 = special.var(ddof=1) / len(special), regular.var(ddof=1) / len(regular)
    se = float(np.sqrt(v1 + v0))
    dof = (v1 + v0) ** 2 / (v1 ** 2 / (len(special)-1) + v0 ** 2 / (len(regular)-1))
    critical = float(stats.t.ppf(.975, dof))
    return {"available": True, "statistic": float(result.statistic), "p_value": float(result.pvalue),
            "regular_n_days": int(len(regular)), "special_n_days": int(len(special)),
            "regular_mean": float(regular.mean()), "special_mean": float(special.mean()),
            "difference": difference, "ci_low": difference-critical*se, "ci_high": difference+critical*se,
            "decision": _decision(float(result.pvalue))}


def weekday_weekend_ttest(frame: pd.DataFrame) -> dict:
    """Supplementary Welch test using one park-wide mean occupancy per date."""
    daily = frame.groupby("date").agg(
        occupancy=("OccupancyRate", "mean"), weekend=("is_weekend", "max")).dropna()
    weekday = daily.loc[~daily.weekend.astype(bool), "occupancy"].to_numpy()
    weekend = daily.loc[daily.weekend.astype(bool), "occupancy"].to_numpy()
    if len(weekday) < 2 or len(weekend) < 2:
        return {"available": False, "reason": "At least two weekday and two weekend dates are needed."}
    result = stats.ttest_ind(weekend, weekday, equal_var=False)
    difference = float(weekend.mean() - weekday.mean())
    v1, v0 = weekend.var(ddof=1) / len(weekend), weekday.var(ddof=1) / len(weekday)
    se = float(np.sqrt(v1 + v0))
    dof = (v1 + v0) ** 2 / (v1 ** 2 / (len(weekend)-1) + v0 ** 2 / (len(weekday)-1))
    critical = float(stats.t.ppf(.975, dof))
    return {"available": True, "statistic": float(result.statistic), "p_value": float(result.pvalue),
            "weekday_n_days": int(len(weekday)), "weekend_n_days": int(len(weekend)),
            "weekday_mean": float(weekday.mean()), "weekend_mean": float(weekend.mean()),
            "difference": difference, "ci_low": difference-critical*se, "ci_high": difference+critical*se,
            "decision": _decision(float(result.pvalue))}


def queue_occupancy_association(frame: pd.DataFrame) -> dict:
    values = frame[["QueueLength", "OccupancyRate"]].dropna()
    if len(values) < 3 or values.QueueLength.nunique() < 2 or values.OccupancyRate.nunique() < 2:
        return {"available": False, "reason": "Insufficient variation for correlation and regression."}
    correlation = stats.pearsonr(values.QueueLength, values.OccupancyRate)
    regression = stats.linregress(values.QueueLength, values.OccupancyRate)
    critical = float(stats.t.ppf(.975, len(values)-2))
    return {"available": True, "n": int(len(values)), "r": float(correlation.statistic),
            "correlation_p": float(correlation.pvalue), "slope": float(regression.slope),
            "intercept": float(regression.intercept), "regression_p": float(regression.pvalue),
            "r_squared": float(regression.rvalue ** 2),
            "slope_ci_low": float(regression.slope - critical*regression.stderr),
            "slope_ci_high": float(regression.slope + critical*regression.stderr),
            "decision": _decision(float(correlation.pvalue))}


def utilization_summary(frame: pd.DataFrame, threshold: float = .80) -> dict:
    rate = frame["OccupancyRate"].dropna()
    per_lot = frame.groupby("SystemCodeNumber").OccupancyRate.agg(
        readings="count", mean="mean", share_above_threshold=lambda s: float((s > threshold).mean()))
    return {"threshold": threshold, "n": int(len(rate)), "above_count": int((rate > threshold).sum()),
            "above_share": float((rate > threshold).mean()), "mean": float(rate.mean()),
            "median": float(rate.median()), "per_lot": per_lot.reset_index().to_dict("records")}


def temporal_anova(frame: pd.DataFrame) -> dict:
    # Aggregate within each date/hour first, avoiding a test sample made of every
    # lot-reading as though all simultaneous readings were independent.
    hourly_daily = frame.groupby(["date", "hour"], as_index=False).OccupancyRate.mean()
    groups = {int(hour): values["OccupancyRate"].dropna().to_numpy()
              for hour, values in hourly_daily.groupby("hour")}
    groups = {hour: values for hour, values in groups.items() if len(values) >= 2}
    if len(groups) < 2:
        return {"available": False, "reason": "Fewer than two hourly groups have enough observations."}
    result = stats.f_oneway(*groups.values())
    hourly = frame.groupby("hour").OccupancyRate.agg(["count", "mean", "std"]).reset_index()
    return {"available": True, "statistic": float(result.statistic), "p_value": float(result.pvalue),
            "hourly": hourly, "decision": _decision(float(result.pvalue))}


def vehicle_type_anova(frame: pd.DataFrame) -> dict:
    groups = {str(name): rows.OccupancyRate.dropna().to_numpy()
              for name, rows in frame.groupby("VehicleType", dropna=True)}
    groups = {name: values for name, values in groups.items() if len(values) >= 2}
    if len(groups) < 2:
        return {"available": False, "reason": "Fewer than two vehicle types have enough observations."}
    result = stats.f_oneway(*groups.values())
    summary = {name: {"n": int(len(values)), "mean": float(values.mean()),
                      "median": float(np.median(values))} for name, values in groups.items()}
    return {"available": True, "statistic": float(result.statistic), "p_value": float(result.pvalue),
            "groups": summary, "decision": _decision(float(result.pvalue))}


def distribution_summary(frame: pd.DataFrame) -> dict:
    output = {}
    for column in ("Occupancy", "QueueLength", "OccupancyRate"):
        values = frame[column].dropna()
        output[column] = {"n": int(values.size), "mean": float(values.mean()),
                          "median": float(values.median()), "variance": float(values.var(ddof=1)),
                          "std": float(values.std(ddof=1)), "skewness": float(values.skew()),
                          "kurtosis": float(values.kurtosis())}
    values = frame.Occupancy.dropna()
    sample = values.sample(n=min(5000, len(values)), random_state=42) if len(values) > 5000 else values
    shapiro = stats.shapiro(sample)
    output["occupancy_normality"] = {"statistic": float(shapiro.statistic),
                                     "p_value": float(shapiro.pvalue), "n_tested": int(len(sample)),
                                     "decision": _decision(float(shapiro.pvalue))}
    outliers = {}
    for column in ("Occupancy", "QueueLength"):
        values = frame[column].dropna()
        q1, q3 = values.quantile([.25, .75])
        iqr = float(q3-q1)
        low, high = float(q1-1.5*iqr), float(q3+1.5*iqr)
        count = int(((values < low) | (values > high)).sum())
        outliers[column] = {"lower_fence": low, "upper_fence": high, "count": count,
                            "percent": count / max(len(values), 1) * 100}
    output["iqr_outliers"] = outliers
    return output


def correlation_analysis(frame: pd.DataFrame) -> dict:
    columns = ["Capacity", "Occupancy", "QueueLength", "OccupancyRate", "hour",
               "day_of_week", "is_weekend", "IsSpecialDay", "traffic_code"]
    values = frame[columns].apply(pd.to_numeric, errors="coerce").dropna(axis=1, how="all")
    matrix = values.corr(method="pearson")
    pairs = [(left, right, float(matrix.loc[left, right]))
             for i, left in enumerate(matrix.columns) for right in matrix.columns[i+1:]
             if pd.notna(matrix.loc[left, right])]
    positive = sorted((p for p in pairs if p[2] > 0), key=lambda p: p[2], reverse=True)[:8]
    negative = sorted((p for p in pairs if p[2] < 0), key=lambda p: p[2])[:8]
    return {"matrix": matrix, "positive": positive, "negative": negative}


def research_insights(frame: pd.DataFrame, traffic: dict, special: dict,
                      queue: dict, temporal: dict) -> list[dict]:
    """Create evidence-led insight cards from observed data and computed tests."""
    cards = []
    if traffic.get("available"):
        groups = traffic["groups"]
        means = {k: v["mean"] for k, v in groups.items()}
        peak_label = max(means, key=means.get)
        cards.append({"title": "Traffic and occupancy", "observation":
                      f"{peak_label.title()} traffic has the highest mean occupancy among observed traffic groups.",
                      "evidence": "; ".join(f"{k}: {v['mean']:.1%} mean (n={v['n']:,})" for k, v in groups.items())
                                  + f"; ANOVA p={traffic['p_value']:.3g}.",
                      "business_meaning": "Traffic condition can inform demand estimates; the observational result does not establish that traffic causes occupancy changes."})
    if special.get("available"):
        direction = "higher" if special["difference"] > 0 else "lower"
        cards.append({"title": "Special-day demand", "observation":
                      f"Daily mean occupancy is {direction} on special days by {abs(special['difference']):.1%}.",
                      "evidence": f"Regular {special['regular_mean']:.1%} vs special {special['special_mean']:.1%}; Welch t-test p={special['p_value']:.3g}.",
                      "business_meaning": "A repeatable difference can justify retaining the special-day flag for forecasts and planning."})
    if queue.get("available"):
        cards.append({"title": "Queue as demand pressure", "observation":
                      f"Queue length and occupancy rate have Pearson r={queue['r']:.3f}.",
                      "evidence": f"n={queue['n']:,}; correlation p={queue['correlation_p']:.3g}; regression slope={queue['slope']:.4f} occupancy-rate units per queued vehicle.",
                      "business_meaning": "Queue length can provide a real-time pressure signal, though correlation alone does not establish causality."})
    if temporal.get("available"):
        hourly = temporal["hourly"]
        low, high = hourly.loc[hourly["mean"].idxmin()], hourly.loc[hourly["mean"].idxmax()]
        cards.append({"title": "Time-of-day pattern", "observation":
                      f"Mean occupancy ranges from {low['mean']:.1%} at {int(low['hour']):02d}:00 to {high['mean']:.1%} at {int(high['hour']):02d}:00.",
                      "evidence": f"One-way ANOVA across observed hours p={temporal['p_value']:.3g}; based on {int(hourly['count'].sum()):,} raw readings.",
                      "business_meaning": "Observed temporal structure supports calendar and lag features for forecasting."})
    utilization = utilization_summary(frame)
    cards.append({"title": "Capacity utilization", "observation":
                  f"{utilization['above_share']:.1%} of readings exceed 80% occupancy rate.",
                  "evidence": f"{utilization['above_count']:,} of {utilization['n']:,} readings; average occupancy rate {utilization['mean']:.1%}.",
                  "business_meaning": "Frequent high utilization leaves less spare capacity and supports evaluating demand-sensitive pricing."})
    return cards
