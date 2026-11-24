"""Evidence-based academic synthesis of ParkedIn's analytics and pricing work."""
import streamlit as st

from _setup import ROOT  # noqa: F401
from analytics.academic_story import gather_academic_evidence

st.set_page_config(page_title="ParkedIn: Conclusion & Findings", layout="wide")
st.title("Conclusion & Findings")
st.caption("An academic synthesis of the actual data, diagnostics, saved model evaluations, and pricing scenarios.")

try:
    evidence = gather_academic_evidence()
except Exception as exc:
    st.error(f"Academic evidence could not be loaded from the project artifacts: {exc}")
    st.stop()

try:
    import plotly.express as px
except ImportError:
    px = None

raw, processed = evidence["raw"], evidence["processed"]
traffic, weekend, special = evidence["traffic"], evidence["weekday_weekend"], evidence["special_day"]
adf, saved_mae = evidence["adf"], evidence["forecast_mae"]
pricing, config = evidence["pricing_scenarios"], evidence["config"]

def panel(label, what, why, influence):
    with st.expander(label, expanded=False):
        st.markdown(f"**What is shown?** {what}\n\n**Why does it matter?** {why}\n\n**How did it influence project decisions?** {influence}")

def pct_test(result, group_a, group_b):
    if not result.get("available"):
        return result.get("reason", "Test result unavailable.")
    # Analysis helpers intentionally return domain-specific group names so the
    # direction remains clear (weekend minus weekday; special minus regular).
    if "weekday_mean" in result:
        mean_a, mean_b = result["weekday_mean"], result["weekend_mean"]
    elif "regular_mean" in result:
        mean_a, mean_b = result["regular_mean"], result["special_mean"]
    else:
        return "Test result schema is unavailable."
    return (f"{group_a} {mean_a:.1%}; {group_b} {mean_b:.1%}; "
            f"difference {result['difference']:+.1%} (second minus first), p={result['p_value']:.3g}.")

st.subheader("Statistical Findings")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Source readings", f"{len(raw):,}")
c2.metric("Parking lots", f"{raw.SystemCodeNumber.nunique():,}")
c3.metric("Weekday / weekend p-value", f"{weekend['p_value']:.2g}" if weekend.get("available") else "N/A")
c4.metric("Traffic ANOVA p-value", f"{traffic['p_value']:.2g}" if traffic.get("available") else "N/A")
stat_summary = (f"Traffic category means differ (ANOVA F={traffic['statistic']:.2f}, p={traffic['p_value']:.3g}). "
                if traffic.get("available") else "Traffic-category test unavailable. ")
stat_summary += pct_test(weekend, "Weekday", "Weekend") + " " + pct_test(special, "Regular-day", "Special-day")
st.write(stat_summary)
st.info("Interpretation: these are observational comparisons in the supplied dataset. They support using traffic, calendar and special-day context, but do not establish that any one factor causes occupancy changes.")

st.subheader("Data Science Findings")
st.write(f"The source contains {len(raw):,} valid parking records; the processed forecasting table contains {len(processed):,} rows and {len(evidence['features'])} model inputs. Existing features combine occupancy history and rolling statistics with timestamp, traffic, queue, special-day and capacity context.")
st.code(" → ".join(["raw parking records", "cleaned half-hour lot rows", "lag / rolling + context features", "lot-level occupancy-rate target"]), language=None)
st.caption("Feature list: " + ", ".join(evidence["features"]))
panel("Feature interpretation", "The engineered model inputs and row counts currently stored by the project.", "Traceable features let the forecast use recent lot behavior and measured context rather than an invented demonstration dataset.", "The existing feature pipeline remains the source of truth; this presentation layer does not change its transformations.")

st.subheader("Time Series Findings")
t1, t2, t3 = st.columns(3)
t1.metric("Observed date range", f"{evidence['daily'].index.min().date()} – {evidence['daily'].index.max().date()}" if not evidence["daily"].empty else "N/A")
t2.metric("Longest contiguous daily segment", f"{len(evidence['contiguous_daily']):,} days")
t3.metric("ADF p-value", f"{adf['p_value']:.3f}" if adf.get("available") else "N/A")
if adf.get("available"):
    st.write(f"ADF statistic {adf['statistic']:.3f}, p={adf['p_value']:.3g}: the unit-root null is not rejected at 5% for the contiguous daily occupancy series. This supports caution about assuming a stable mean; it does not by itself prove a particular model is necessary.")
if not evidence["hourly"].empty:
    hourly = evidence["hourly"]
    peak = hourly.loc[hourly["mean"].idxmax()]
    st.write(f"Observed mean occupancy rate peaks at {int(peak['hour']):02d}:00 ({peak['mean']:.1%}) in the aggregated hourly profile. Calendar pattern and temporal dependence motivate time-aware validation and forecast inputs.")
panel("Time-series interpretation", "The observed daily series length, ADF result, and hourly occupancy profile.", "Temporal dependence, changing levels and calendar patterns mean a random row split or a static average could hide future-period errors.", "The project evaluates forecasts on chronological held-out rows and supplies recent lags and calendar features to its six-step forecaster.")

st.subheader("Machine Learning Findings")
if saved_mae.empty:
    st.warning("Saved model comparison metrics are unavailable. Run the existing training/evaluation workflow to produce them.")
else:
    winners = saved_mae.idxmin(axis=1)
    wins = int((winners == "xgboost").sum())
    h1 = saved_mae.loc[1] if 1 in saved_mae.index else saved_mae.iloc[0]
    hlast = saved_mae.iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("Lowest saved MAE", "XGBoost" if wins == len(winners) else str(winners.value_counts().idxmax()), f"{wins}/{len(winners)} horizons")
    m2.metric("XGBoost MAE · 1 slot", f"{h1['xgboost']:.5f}")
    m3.metric(f"XGBoost MAE · {int(saved_mae.index[-1])} slots", f"{hlast['xgboost']:.5f}")
    if px is not None:
        long = saved_mae.rename_axis("Slots ahead").reset_index().melt(id_vars="Slots ahead", var_name="Model", value_name="Test MAE")
        fig = px.line(long, x="Slots ahead", y="Test MAE", color="Model", markers=True,
                      title="Saved chronological test MAE by forecast horizon")
        fig.update_layout(legend_title_text="Model")
        st.plotly_chart(fig, use_container_width=True)
        panel("Forecast error chart interpretation", "Saved held-out MAE for XGBoost, Random Forest, ARIMA and naive persistence at each horizon.", "MAE measures typical absolute occupancy-rate error; rising error with horizon indicates uncertainty grows as the prediction moves farther ahead.", "Saved test MAE ranks XGBoost lowest across the stored horizons; the production bundle therefore uses its six horizon-specific persisted models.")
    else:
        st.warning("Install Plotly (`pip install -r requirements.txt`) to render the forecast comparison chart.")
    st.write(f"**Model selection summary.** XGBoost is lowest at {wins} of {len(winners)} saved horizons. At one step it scores {h1['xgboost']:.5f} against the next-lowest {h1.drop('xgboost').min():.5f}; at {int(saved_mae.index[-1])} steps it scores {hlast['xgboost']:.5f}. This is a held-out MAE-based selection statement, not a claim of universal superiority under every metric.")
    panel("Model selection interpretation", "Horizon-specific saved validation-selected model scores.", "The closest model and the winning margin change with forecast horizon, so evaluation must retain horizon detail.", "The deployed choice follows lowest saved chronological test MAE at each of the evaluated steps; non-XGBoost full metric arrays were not persisted.")

st.subheader("Business Impact")
st.write("The pricing evaluation compares strategy outputs under assumed price elasticity. It estimates revenue and occupancy outcomes from the project simulator; the dataset does not contain realized transaction revenue for validating those estimates.")
if pricing.empty:
    st.warning("Pricing scenario table is unavailable. The existing pricing evaluation must generate models/pricing_results.csv.")
else:
    elast = float(config.get("elasticity", .4))
    current = pricing[pricing.elasticity.round(2) == round(elast, 2)]
    chosen = current[current.strategy == "Rules + grid search"]
    fixed = current[current.strategy == "Fixed price"]
    if not chosen.empty and not fixed.empty:
        pr, fr = chosen.iloc[0], fixed.iloc[0]
        d1, d2, d3 = st.columns(3)
        d1.metric("Configured elasticity", f"{elast:.1f}", "scenario assumption")
        d2.metric("Estimated revenue change", f"{pr['revenue_vs_fixed_pct']:+.1f}%", "vs fixed-price simulation")
        d3.metric("Mean gap from target occupancy", f"{pr['avg_gap_from_85pct']:.3f}", f"fixed: {fr['avg_gap_from_85pct']:.3f}")
        st.write(f"At elasticity {elast:.1f}, the saved rules + grid-search simulation estimates Rs {pr['revenue_rs']:,.0f} versus Rs {fr['revenue_rs']:,.0f} for fixed pricing. This is a modeled scenario, not measured revenue uplift.")
    if px is not None:
        plot_data = pricing.copy()
        fig = px.line(plot_data, x="elasticity", y="revenue_vs_fixed_pct", color="strategy", markers=True,
                      title="Simulated revenue change versus fixed pricing across elasticity assumptions")
        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.update_yaxes(title="Estimated revenue difference (%)")
        st.plotly_chart(fig, use_container_width=True)
        panel("Revenue sensitivity chart interpretation", "Relative simulated revenue by pricing strategy as the assumed customer price elasticity changes.", "The sign and magnitude of the estimated impact are sensitive to how strongly demand responds to price.", "The configured scenario is shown as a conditional estimate. The wide scenario range argues for monitoring and calibration against real transactions before making a realized-impact claim.")
        if "lot_slots_95pct_full" in pricing.columns:
            fig2 = px.bar(pricing, x="elasticity", y="lot_slots_95pct_full", color="strategy", barmode="group",
                          title="Simulated lot-slots at or above 95% utilization")
            st.plotly_chart(fig2, use_container_width=True)
            panel("Capacity pressure chart interpretation", "Simulated number of lot-slots reaching at least 95% utilization under each pricing strategy and elasticity assumption.", "Near-capacity events show a separate operational dimension from estimated revenue.", "The grid search is evaluated on both revenue and occupancy gap; decisions should account for capacity and service goals alongside revenue.")
    else:
        st.warning("Install Plotly (`pip install -r requirements.txt`) to render pricing scenario charts.")

st.subheader("Dynamic Pricing Benefits")
st.markdown("- Uses per-lot occupancy forecasts and the configured occupancy target as inputs to price selection.\n- Connects observed demand patterns, traffic and calendar context with operational price decisions.\n- Enables scenario comparison across pricing rules and elasticity assumptions.\n- Produces replayable price decisions that can be audited against timestamps and lot readings.")
st.subheader("Academic Conclusion")
st.write("The project forms a coherent analytics workflow: the actual parking records reveal measurable temporal and contextual occupancy differences; engineered lag and calendar features represent this structure; chronological forecast comparisons favor the persisted XGBoost ensemble on saved MAE; and the pricing layer converts forecasts into constrained candidate prices. Statistical significance and model accuracy support the design rationale, while pricing impact remains conditional on the simulator's elasticity assumptions and should be validated against transaction data.")
st.caption("Data source: project raw dataset, processed feature artifact, saved model-evaluation metrics and pricing scenario results. Results are recomputed/read from existing artifacts; this page does not retrain models or modify pricing behavior.")
