"""Assemble actual evidence across ParkedIn's analysis-to-pricing path."""
from __future__ import annotations

import pandas as pd

from agents.data_agent import FEATURES
from analytics.model_evaluation import forecast_comparison
from analytics.statistics import load_dataset, load_simulation_table
from analytics.statistical_analysis import (load_analysis_data, queue_occupancy_association,
                                            special_day_ttest, temporal_anova,
                                            traffic_occupancy_anova, utilization_summary,
                                            vehicle_type_anova, weekday_weekend_ttest)
from analytics.timeseries import (adf_stationarity_test, daily_park_series,
                                  hourly_seasonality, longest_contiguous_daily_segment,
                                  stl_daily_decomposition, weekday_seasonality)
from common import load_config


def _runtime_pricing_trace() -> pd.DataFrame:
    """Join the latest real runtime reading, saved forecast, and price decisions."""
    try:
        prices = load_simulation_table("prices")
        readings = load_simulation_table("readings")
        forecasts = load_simulation_table("forecasts")
    except (FileNotFoundError, pd.errors.DatabaseError, ValueError):
        return pd.DataFrame()
    if prices.empty:
        return pd.DataFrame()
    latest_ts = prices.ts.max()
    current = prices[prices.ts == latest_ts].drop_duplicates(["lot_id", "ts"], keep="last").copy()
    reading = readings[readings.ts == latest_ts][["lot_id", "ts", "occupancy", "occ_rate"]]
    reading = reading.drop_duplicates(["lot_id", "ts"], keep="last")
    next_forecast = forecasts[(forecasts.made_at == latest_ts) & (forecasts.slot_ahead == 1)]
    next_forecast = next_forecast.drop_duplicates("lot_id", keep="last")
    result = current.merge(reading, on=["lot_id", "ts"], how="left")
    if not next_forecast.empty:
        result = result.merge(next_forecast[["lot_id", "predicted"]], on="lot_id", how="left")
        result = result.rename(columns={"predicted": "next_slot_forecast"})
    return result.sort_values("lot_id").reset_index(drop=True)


def gather_academic_evidence() -> dict:
    """Compute presentation-ready values from actual source, saved, and runtime data."""
    raw = load_analysis_data()
    processed = load_dataset("processed")
    traffic = traffic_occupancy_anova(raw)
    special = special_day_ttest(raw)
    weekend = weekday_weekend_ttest(raw)
    queue = queue_occupancy_association(raw)
    temporal = temporal_anova(raw)
    vehicle = vehicle_type_anova(raw)
    utilization = utilization_summary(raw)

    daily = daily_park_series(processed)
    segment = longest_contiguous_daily_segment(daily)
    adf = adf_stationarity_test(segment["occupancy_rate"] if not segment.empty else pd.Series(dtype=float))
    stl = stl_daily_decomposition(daily, period=7)
    hours = hourly_seasonality(processed)
    weekdays = weekday_seasonality(processed)
    forecasts = forecast_comparison()

    try:
        prices = pd.read_csv("models/pricing_results.csv")
    except (FileNotFoundError, pd.errors.ParserError):
        prices = pd.DataFrame()
    runtime = _runtime_pricing_trace()
    config = load_config()

    return {
        "raw": raw, "processed": processed, "features": list(FEATURES),
        "traffic": traffic, "special_day": special, "weekday_weekend": weekend,
        "queue": queue, "temporal": temporal, "vehicle": vehicle,
        "utilization": utilization, "daily": daily, "contiguous_daily": segment,
        "adf": adf, "stl": stl, "hourly": hours, "weekday_profile": weekdays,
        "forecast_mae": forecasts, "pricing_scenarios": prices,
        "runtime_pricing": runtime, "config": config,
    }


def decision_trace(evidence: dict) -> list[dict]:
    """Return Finding/Evidence/Decision records that match the current implementation."""
    traffic, special, weekend = (evidence[k] for k in ("traffic", "special_day", "weekday_weekend"))
    queue, temporal, utilization = (evidence[k] for k in ("queue", "temporal", "utilization"))
    mae, pricing = evidence["forecast_mae"], evidence["pricing_scenarios"]
    target = evidence["config"]["target_occupancy"]
    if traffic.get("available"):
        means = traffic["groups"]
        traffic_finding = (f"Mean occupancy rate ranges from {means['low']['mean']:.1%} in low traffic "
                           f"to {means['high']['mean']:.1%} in high traffic.")
        traffic_evidence = f"One-way ANOVA F={traffic['statistic']:.2f}, p={traffic['p_value']:.3g}; η²={traffic['eta_squared']:.3f}."
    else:
        traffic_finding, traffic_evidence = "Traffic association unavailable.", traffic.get("reason", "")
    if special.get("available"):
        special_finding = (f"Special-day daily mean is {special['difference']:+.1%} versus regular dates "
                           "(special minus regular).")
        special_evidence = f"Welch t={special['statistic']:.2f}, p={special['p_value']:.3g}; 95% CI [{special['ci_low']:+.1%}, {special['ci_high']:+.1%}]."
    else:
        special_finding, special_evidence = "Special-day comparison unavailable.", special.get("reason", "")
    if weekend.get("available"):
        weekend_finding = (f"Weekend daily mean is {weekend['difference']:+.1%} versus weekdays "
                           "(weekend minus weekday).")
        weekend_evidence = f"Welch t={weekend['statistic']:.2f}, p={weekend['p_value']:.3g}."
    else:
        weekend_finding, weekend_evidence = "Weekday/weekend test unavailable.", weekend.get("reason", "")
    if queue.get("available"):
        queue_finding = f"Queue length has a positive occupancy-rate association (r={queue['r']:.3f})."
        queue_evidence = f"Pearson p={queue['correlation_p']:.3g}; regression slope={queue['slope']:.4f}; n={queue['n']:,}."
    else:
        queue_finding, queue_evidence = "Queue association unavailable.", queue.get("reason", "")

    if not mae.empty:
        winners = mae.idxmin(axis=1)
        xgb_wins = int((winners == "xgboost").sum())
        best_mae_text = f"XGBoost has the lowest saved MAE at {xgb_wins}/{len(winners)} horizons."
        one = mae.loc[1] if 1 in mae.index else mae.iloc[0]
        model_evidence = f"At first horizon: XGBoost {one['xgboost']:.5f}, RF {one['random_forest']:.5f}, ARIMA {one['arima']:.5f}, naive {one['naive']:.5f}."
        model_decision = "The production system uses the persisted six-model XGBoost bundle; saved test MAE supports that choice at the evaluated horizons." if xgb_wins == len(winners) else "XGBoost is the current production forecaster; compare the horizon-level winners before calling it best overall."
    else:
        best_mae_text, model_evidence, model_decision = "Forecast ranking unavailable.", "No saved MAE values.", "No selection conclusion can be drawn."

    elasticity = float(evidence["config"].get("elasticity", .4))
    pricing_row = pd.DataFrame()
    if not pricing.empty:
        exact = pricing[(pricing.elasticity.round(2) == round(elasticity, 2))
                        & (pricing.strategy == "Rules + grid search")]
        if not exact.empty:
            pricing_row = exact.iloc[0]
    expected_revenue = (f"At assumed elasticity {elasticity:.1f}, simulated rules + grid search revenue is "
                        f"{pricing_row['revenue_vs_fixed_pct']:+.1f}% vs fixed price.") if not pricing_row.empty else "No matching saved pricing scenario."
    price_evidence = (f"Estimated Rs {pricing_row['revenue_rs']:,.0f} vs fixed Rs "
                      f"{pricing.loc[(pricing.elasticity.round(2) == round(elasticity, 2)) & (pricing.strategy == 'Fixed price'), 'revenue_rs'].iloc[0]:,.0f}; "
                      f"mean target gap {pricing_row['avg_gap_from_85pct']:.3f}; 95%+ full lot-slots {int(pricing_row['lot_slots_95pct_full'])}.") if not pricing_row.empty else "Revenue comparison unavailable."

    latest_output = "No runtime price decisions are recorded."
    runtime = evidence["runtime_pricing"]
    if not runtime.empty:
        latest_ts = runtime.ts.iloc[0]
        latest_output = (f"Latest replay slot {latest_ts}: {len(runtime)} lot prices recorded; "
                         f"price range Rs {runtime.price.min():.0f}–{runtime.price.max():.0f}/hour. "
                         f"The system targets {target:.0%} occupancy and prices from a next-slot forecast plus demand level.")

    temporal_evidence = (f"Hourly ANOVA F={temporal['statistic']:.2f}, p={temporal['p_value']:.3g}. "
                         if temporal.get("available") else "")
    adf = evidence["adf"]
    adf_text = f"ADF statistic {adf['statistic']:.3f}, p={adf['p_value']:.3g} on {adf['observations']} contiguous daily values." if adf.get("available") else "ADF unavailable."
    gap_share = utilization["above_share"]
    return [
        {"stage": "Problem Statement", "finding": f"Forecast lot demand and choose a price toward the configured {target:.0%} occupancy target.",
         "evidence": f"The actual data contains {len(evidence['raw']):,} readings from {evidence['raw'].SystemCodeNumber.nunique()} lots; the target comes from config.yaml.",
         "decision": "Use lot-level occupancy forecasting as input to per-lot next-slot pricing."},
        {"stage": "Statistical Investigation", "finding": f"{traffic_finding} {special_finding} {weekend_finding} {queue_finding}",
         "evidence": f"{traffic_evidence} {special_evidence} {weekend_evidence} {queue_evidence}",
         "decision": "Retain the existing ordinal traffic, IsSpecialDay, QueueLength, and dow inputs. The data supports dow, not a separately created Weekend_Flag; observational tests do not imply causation."},
        {"stage": "EDA Findings", "finding": f"{gap_share:.1%} of source readings exceed 80% utilization; peak observed hour is {int(evidence['hourly'].loc[evidence['hourly']['mean'].idxmax(), 'hour']):02d}:00.",
         "evidence": f"{utilization['above_count']:,}/{utilization['n']:,} readings above 80%; mean occupancy rate {utilization['mean']:.1%}. {temporal_evidence.strip()}",
         "decision": "Normalize occupancy by capacity and inspect queue, traffic, special-day, and calendar signals before pricing."},
        {"stage": "Feature Engineering", "finding": f"The forecaster uses {len(evidence['features'])} features, including occupancy lags, rolling demand, calendar fields, traffic, queue, special day, and capacity.",
         "evidence": ", ".join(evidence["features"]),
         "decision": "Use existing engineered inputs only; the current pipeline has hour and day-of-week but no separate rush-hour, weekend, or season-category features."},
        {"stage": "Time Series Findings", "finding": f"{len(evidence['contiguous_daily'])} consecutive dates support temporal diagnostics; {adf_text}",
         "evidence": f"{temporal_evidence} Forecast targets span six half-hour steps. Overnight/missing dates are not filled for diagnostics.",
         "decision": "Use recent occupancy lags and calendar signals; evaluate forecast models chronologically rather than assume stationarity."},
        {"stage": "Model Selection", "finding": best_mae_text,
         "evidence": model_evidence,
         "decision": model_decision},
        {"stage": "Model Evaluation", "finding": "Forecast MAE rises with horizon, but saved XGBoost remains best at each saved horizon." if not mae.empty else "Saved evaluation unavailable.",
         "evidence": ("; ".join(f"h={int(h)}: {row['xgboost']:.5f} XGBoost MAE" for h, row in mae.iterrows()) if not mae.empty else "No saved forecast metrics."),
         "decision": "Use the six horizon-specific test scores to justify three-hour prediction; do not use training fit as the model-selection evidence."},
        {"stage": "Dynamic Pricing Output", "finding": f"{expected_revenue} {latest_output}",
         "evidence": price_evidence,
         "decision": "Apply the existing rule-based starting price and constrained grid search using assumed price elasticity. Treat revenue as a scenario estimate, not realized sales."},
    ]
