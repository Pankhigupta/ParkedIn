"""Temporal analysis using processed project data and persisted forecasts."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from _setup import ROOT  # noqa: F401  # adds root for project imports
from agents.data_agent import FEATURES
from agents.forecast_agent import ForecastAgent
from analytics.model_evaluation import forecast_comparison
from analytics.statistics import load_dataset, load_simulation_table
from analytics.timeseries import (adf_stationarity_test, daily_park_series,
                                  hourly_seasonality, longest_contiguous_daily_segment,
                                  park_wide_series, stl_daily_decomposition,
                                  weekday_seasonality)
from common import FORECAST_MODEL_PATH, METRICS_PATH

st.set_page_config(page_title="ParkedIn: Time Series Analysis", layout="wide")
st.title("Time Series Analysis")
st.caption("Inspect observed demand structure, model assumptions, and held-out forecasts from the existing pipeline.")


@st.cache_data(show_spinner="Loading processed project data…")
def load_processed_data():
    return load_dataset("processed")


@st.cache_resource(show_spinner="Loading saved XGBoost forecasters…")
def load_forecaster():
    return ForecastAgent(horizon=6)


try:
    data = load_processed_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Processed project dataset unavailable: {exc}")
    st.stop()

series = park_wide_series(data)
daily = daily_park_series(data)
hour_profile = hourly_seasonality(data)
weekday_profile = weekday_seasonality(data)
regular_daily = longest_contiguous_daily_segment(daily)
adf_result = adf_stationarity_test(regular_daily["occupancy_rate"] if not regular_daily.empty else pd.Series(dtype=float))
decomposition = stl_daily_decomposition(daily, period=7)
evaluation = forecast_comparison()

st.markdown("**Analysis data:** existing engineered feature dataset (`data/processed/features.pkl`). "
            "Park-wide occupancy rate is total occupancy divided by total capacity. Processed lot-slot values include the existing interpolation for missing readings.")
st.caption(f"{len(data):,} processed lot-slot rows · {data.SystemCodeNumber.nunique()} lots · "
           f"{series.index.min():%d %b %Y %H:%M} to {series.index.max():%d %b %Y %H:%M} · "
           f"{len(series):,} observed half-hour timestamps · overnight gaps are not filled.")

tabs = st.tabs(["1. Time Series Overview", "2. Trend Analysis", "3. Seasonality Analysis",
                "4. Stationarity Analysis", "5. Decomposition", "6. Autocorrelation",
                "7. Forecasting Justification"])


# 1. Time series overview
with tabs[0]:
    st.subheader("Occupancy and demand over time")
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(series.index, series.occupancy, color="#4267ac", linewidth=1)
    axes[0].set(title="Total occupied spaces across lots", ylabel="Occupied spaces")
    axes[1].plot(series.index, series.occupancy_rate, color="#258277", linewidth=1)
    axes[1].set(title="Capacity-weighted occupancy rate (demand level)", ylabel="Occupancy rate", xlabel="Observed time")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    st.caption("The project dataset contains daytime lot readings. Overnight and missing-date gaps remain gaps; this page does not synthesize readings.")

    st.markdown("#### Revenue over time")
    try:
        metrics = load_simulation_table("metrics")
        revenue_rows = metrics[(metrics.lot_id == "ALL")
                               & metrics.name.isin(["revenue_dynamic", "revenue_fixed"])]
        if revenue_rows.empty:
            st.info("No runtime revenue metrics are recorded. The source parking dataset has no price or revenue observations, so revenue cannot be reconstructed from it.")
        else:
            revenue_rows = revenue_rows.copy()
            revenue_rows["ts"] = pd.to_datetime(revenue_rows.ts)
            revenue = revenue_rows.pivot_table(index="ts", columns="name", values="value", aggfunc="sum").sort_index()
            chart = pd.DataFrame(index=revenue.index)
            if "revenue_dynamic" in revenue:
                chart["Dynamic pricing estimate"] = revenue["revenue_dynamic"].cumsum()
            if "revenue_fixed" in revenue:
                chart["Fixed-price estimate"] = revenue["revenue_fixed"].cumsum()
            st.line_chart(chart, y_label="Cumulative simulated revenue (Rs)")
            st.caption("These runtime metrics are simulator revenue estimates based on assumed price elasticity, not observed transaction revenue.")
    except (FileNotFoundError, ValueError, pd.errors.DatabaseError) as exc:
        st.info(f"Revenue history unavailable: {exc}. The raw dataset contains no revenue or price fields.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Mean occupancy rate", f"{series.occupancy_rate.mean():.1%}")
    c2.metric("Peak observed slot rate", f"{series.occupancy_rate.max():.1%}")
    c3.metric("Typical slot interval", "30 minutes")


# 2. Trend analysis
with tabs[1]:
    st.subheader("Trend analysis")
    st.write("Daily park-wide means use available processed values. The descriptive linear trend uses actual calendar dates and does not fill missing dates.")
    trend_data = daily.dropna(subset=["occupancy_rate"]).copy()
    slope = 0.0
    if len(trend_data) >= 3:
        x = (trend_data.index - trend_data.index.min()).days.to_numpy(dtype=float)
        slope, intercept = np.polyfit(x, trend_data.occupancy_rate.to_numpy(), 1)
        trend_data["trend"] = intercept + slope*x
        fig, ax = plt.subplots(figsize=(11, 4.5))
        ax.plot(trend_data.index, trend_data.occupancy_rate, marker=".", linewidth=1,
                color="#4267ac", label="Daily mean")
        ax.plot(trend_data.index, trend_data.trend, color="#c04b42", linewidth=2,
                label="Linear trend across recorded dates")
        ax.set(title="Daily mean occupancy rate and trend", xlabel="Date", ylabel="Occupancy rate")
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.metric("Trend slope", f"{slope*100:+.3f} occupancy-rate percentage points per calendar day")
    else:
        st.info("Not enough dates to estimate a trend.")
    daily_table = daily.reset_index()
    st.dataframe(daily_table.rename(columns={"date": "Date", "occupancy": "Mean occupied spaces",
                                             "occupancy_rate": "Mean occupancy rate",
                                             "queue_length": "Mean queue length", "slots": "Processed slots"}),
                 hide_index=True, use_container_width=True)


# 3. Seasonality analysis
with tabs[2]:
    st.subheader("Seasonality analysis")
    st.write("These profiles summarize observed within-day and weekday patterns. They do not create rush-hour or season categories.")
    c1, c2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(hour_profile.hour, hour_profile["mean"], marker="o", color="#258277")
    band = hour_profile["std"].fillna(0)
    ax.fill_between(hour_profile.hour, (hour_profile["mean"]-band).clip(0, 1),
                    (hour_profile["mean"]+band).clip(0, 1), alpha=.18, color="#258277")
    ax.set(title="Mean occupancy by hour", xlabel="Hour of day", ylabel="Occupancy rate", ylim=(0, 1))
    c1.pyplot(fig, clear_figure=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(weekday_profile.day_name, weekday_profile["mean"], color="#698bb7")
    ax.set(title="Mean occupancy by day of week", xlabel="Day", ylabel="Occupancy rate", ylim=(0, 1))
    ax.tick_params(axis="x", rotation=35)
    c2.pyplot(fig, clear_figure=True)
    st.caption("Coverage is roughly 08:00–16:30. Nighttime and annual seasonality cannot be inferred from these data.")
    st.dataframe(weekday_profile.rename(columns={"day_name": "Day", "count": "Processed slots",
                                                  "mean": "Mean occupancy rate", "std": "Std. deviation"}),
                 hide_index=True, use_container_width=True)


# 4. Stationarity analysis
with tabs[3]:
    st.subheader("Stationarity analysis · Augmented Dickey–Fuller")
    st.write("The ADF test uses daily park-wide occupancy rates from the longest uninterrupted calendar segment. Missing dates are excluded, not interpolated.")
    if adf_result.get("available"):
        c1, c2, c3 = st.columns(3)
        c1.metric("ADF statistic", f"{adf_result['statistic']:.4f}")
        c2.metric("p-value", f"{adf_result['p_value']:.4g}")
        c3.metric("Test observations", f"{adf_result['observations']}")
        st.write(f"Segment: {regular_daily.index.min():%d %b %Y}–{regular_daily.index.max():%d %b %Y} · lags used: {adf_result['used_lags']}")
        with st.expander("ADF critical values"):
            st.json(adf_result["critical_values"])
        if adf_result["is_stationary_at_5pct"]:
            st.success("At α=0.05, reject the unit-root null for this segment; the evidence is consistent with stationarity in this sample.")
        else:
            st.warning("At α=0.05, fail to reject the unit-root null; this segment does not provide sufficient evidence of stationarity.")
        st.info("The primary XGBoost forecaster does not require a stationary series. Stationarity matters more directly to the ARIMA evaluation baseline.")
    else:
        st.info(adf_result.get("reason", "ADF unavailable."))


# 5. Decomposition
with tabs[4]:
    st.subheader("STL decomposition")
    if decomposition.get("available"):
        components = decomposition["components"]
        fig, axes = plt.subplots(4, 1, figsize=(11, 9), sharex=True)
        for ax, field, title, color in zip(
                axes, ["observed", "trend", "seasonal", "residual"],
                ["Observed daily occupancy rate", "Trend", "Period-7 seasonal component", "Residual"],
                ["#4267ac", "#c04b42", "#258277", "#777777"]):
            ax.plot(components.index, components[field], color=color, linewidth=1.3)
            ax.set_title(title, loc="left", fontsize=10)
            ax.set_ylabel(field.title())
        axes[-1].set_xlabel("Date")
        fig.suptitle("STL components · daily park-wide occupancy rate", y=1.01)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption(f"STL period={decomposition['period']}; longest uninterrupted calendar segment "
                   f"{decomposition['start']:%d %b %Y}–{decomposition['end']:%d %b %Y} "
                   f"(n={decomposition['observations']}). Missing dates were not filled.")
        st.write(f"Seasonal component SD: {components.seasonal.std():.4f} · residual SD: {components.residual.std():.4f} occupancy-rate units.")
    else:
        st.info(decomposition.get("reason", "STL decomposition unavailable."))


# 6. Autocorrelation
with tabs[5]:
    st.subheader("Autocorrelation · ACF and PACF")
    st.write("Both plots use the same contiguous daily occupancy-rate segment used for ADF and STL.")
    if len(regular_daily) >= 10:
        values = regular_daily.occupancy_rate.astype(float)
        lags = min(14, len(values)//2-1)
        left, right = st.columns(2)
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_acf(values, lags=lags, ax=ax, title="ACF · daily occupancy rate")
        left.pyplot(fig, clear_figure=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        plot_pacf(values, lags=lags, method="ywm", ax=ax, title="PACF · daily occupancy rate")
        right.pyplot(fig, clear_figure=True)
        st.caption(f"{len(values)} consecutive processed daily values · lags 1–{lags}. Correlation bands are approximate and do not identify causal effects.")
    else:
        st.info("At least 10 consecutive daily values are needed for ACF/PACF.")


# 7. Forecasting justification
with tabs[6]:
    st.subheader("Forecasting justification")
    st.write("Repeated half-hour occupancy readings, time-of-day and weekday patterns, and lag dependence provide information for anticipating near-future occupancy. Forecasting helps the pricing layer use expected next-slot demand rather than only the latest reading.")
    if not evaluation.empty:
        st.markdown("#### Existing model comparison · saved chronological test MAE")
        st.dataframe(evaluation.style.format("{:.4f}"), use_container_width=True)
        st.caption("The saved evaluation compares naive, ARIMA, Random Forest, and XGBoost occupancy forecasts. Lower MAE is better; no model is retrained by this page.")
    elif METRICS_PATH.is_file():
        st.info("The metrics file exists but has no forecast comparison table.")
    else:
        st.info("No saved evaluation metrics were found. Run the existing training workflow to create them; this page does not train models.")

    st.markdown("#### Existing XGBoost · selected held-out forecast trace")
    if not FORECAST_MODEL_PATH.is_file():
        st.info("Saved forecast models are not present, so a trace cannot be generated.")
    else:
        try:
            eligible = data[(data.split == "test") & data[FEATURES].notna().all(axis=1)]
            lots = sorted(eligible.SystemCodeNumber.unique().tolist())
            if not lots:
                st.info("No complete test rows are available for the saved model.")
            else:
                lot = st.selectbox("Parking lot", lots, key="tsa_lot")
                lot_data = data[data.SystemCodeNumber == lot].sort_values("slot").reset_index(drop=True)
                origins = []
                for i in range(len(lot_data)-6):
                    row = lot_data.iloc[i]
                    if row.split != "test" or row[FEATURES].isna().any():
                        continue
                    future = lot_data.iloc[i+1:i+7]
                    expected = pd.date_range(row.slot + pd.Timedelta(minutes=30), periods=6, freq="30min")
                    if len(future) == 6 and np.array_equal(future.slot.to_numpy(), expected.to_numpy()) \
                            and (future.split == "test").all():
                        origins.append(pd.Timestamp(row.slot))
                if not origins:
                    st.info("This lot has no complete six-slot, half-hourly forecast window in the test period.")
                else:
                    origin = st.selectbox("Test-set forecast origin", origins, index=len(origins)-1,
                                          format_func=lambda value: value.strftime("%d %b %Y, %H:%M"),
                                          key="tsa_origin")
                    origin_row = lot_data[lot_data.slot == origin].iloc[[0]]
                    forecast = np.asarray(load_forecaster().predict(origin_row)[lot], dtype=float)
                    future_times = [origin + pd.Timedelta(minutes=30*h) for h in range(1, 7)]
                    actual = lot_data.set_index("slot").loc[future_times, "occ_rate"].to_numpy(dtype=float)
                    error = float(np.mean(np.abs(forecast-actual)))
                    history = lot_data[(lot_data.slot <= origin) & (lot_data.slot >= origin-pd.Timedelta(hours=8))]
                    fig, ax = plt.subplots(figsize=(10, 4.5))
                    ax.plot(history.slot, history.occ_rate, marker=".", color="#4267ac", label="Processed history")
                    ax.plot(future_times, actual, marker="o", color="#222222", label="Held-out actual")
                    ax.plot(future_times, forecast, marker="o", linestyle="--", color="#ca5b48", label="Saved XGBoost forecast")
                    ax.axvline(origin, color="#777777", linestyle=":")
                    ax.set(title=f"Six-step forecast · {lot}", xlabel="Time", ylabel="Occupancy rate", ylim=(0, 1))
                    ax.legend()
                    fig.tight_layout()
                    st.pyplot(fig, clear_figure=True)
                    st.metric("MAE for this selected six-slot window", f"{error:.4f}")
                    st.caption("One selected held-out example; use the aggregate test-set MAE table for overall model comparison.")
        except Exception as exc:
            st.warning(f"Saved forecast trace unavailable: {exc}")

    st.info("The production forecast uses six saved XGBoost models, one per half-hour step up to three hours. ARIMA, Random Forest, and naive forecasts are evaluation comparisons. The pricing layer uses the next-slot occupancy forecast.")


st.divider()
st.subheader("Automated TSA Findings Summary")
findings = [f"The processed dataset contains {len(series):,} park-wide half-hour timestamps from {series.index.min():%d %b %Y} to {series.index.max():%d %b %Y}; overnight and missing-date gaps are not filled."]
if len(trend_data) >= 3:
    direction = "increasing" if slope > 0 else "decreasing" if slope < 0 else "approximately flat"
    findings.append(f"The descriptive trend is {direction}: {slope*100:+.3f} occupancy-rate percentage points per calendar day across recorded dates.")
if not hour_profile.empty:
    peak, low = hour_profile.loc[hour_profile["mean"].idxmax()], hour_profile.loc[hour_profile["mean"].idxmin()]
    findings.append(f"Mean occupancy peaks at {int(peak.hour):02d}:00 ({peak['mean']:.1%}) and is lowest at {int(low.hour):02d}:00 ({low['mean']:.1%}) among observed hours.")
if not weekday_profile.empty:
    peak_day = weekday_profile.loc[weekday_profile["mean"].idxmax()]
    findings.append(f"The highest observed day-of-week mean is {peak_day.day_name} ({peak_day['mean']:.1%}).")
if adf_result.get("available"):
    action = "rejects" if adf_result["is_stationary_at_5pct"] else "does not reject"
    findings.append(f"ADF {action} the unit-root null on the longest regular daily segment (p={adf_result['p_value']:.4g}).")
if decomposition.get("available"):
    findings.append(f"STL separated trend, period-7 seasonal, and residual components over {decomposition['observations']} consecutive dates.")
if not evaluation.empty and "xgboost" in evaluation.columns:
    row = evaluation.loc[1] if 1 in evaluation.index else evaluation.iloc[0]
    if "naive" in row and pd.notna(row["naive"]):
        relation = "lower" if row["xgboost"] < row["naive"] else "not lower"
        findings.append(f"At one slot ahead, saved XGBoost test MAE ({row['xgboost']:.4f}) is {relation} than naive MAE ({row['naive']:.4f}).")
findings.append("Revenue is only available as a runtime simulator estimate; the source dataset contains no observed prices or transaction revenue.")
for finding in findings:
    st.markdown(f"- {finding}")
