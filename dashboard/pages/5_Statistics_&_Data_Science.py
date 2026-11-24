"""Statistical investigation of the actual ParkedIn source dataset."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

# _setup adds the project root to sys.path. This page intentionally does not call
# check_ready: its analyses depend only on the source dataset, not the live backend.
from _setup import ROOT  # noqa: F401
from analytics.statistical_analysis import (
    correlation_analysis,
    distribution_summary,
    load_analysis_data,
    queue_occupancy_association,
    research_insights,
    special_day_ttest,
    temporal_anova,
    traffic_occupancy_anova,
    utilization_summary,
    vehicle_type_anova,
    weekday_weekend_ttest,
)
st.set_page_config(page_title="ParkedIn: Statistics & Data Science", layout="wide")
st.title("Statistics & Data Science")
st.caption("Evidence from the project's observed parking dataset, connected to feature and model decisions.")


@st.cache_data(show_spinner="Loading the project dataset…")
def cached_analysis_data():
    return load_analysis_data()


df = cached_analysis_data()
traffic_result = traffic_occupancy_anova(df)
special_result = special_day_ttest(df)
queue_result = queue_occupancy_association(df)
temporal_result = temporal_anova(df)
vehicle_result = vehicle_type_anova(df)
weekend_result = weekday_weekend_ttest(df)
utilization_result = utilization_summary(df)
distribution_result = distribution_summary(df)
correlation_result = correlation_analysis(df)
insight_cards = research_insights(df, traffic_result, special_result, queue_result, temporal_result)

st.markdown("**Analysis path**")
st.markdown("`Raw parking readings` → `Statistical investigation` → `EDA` → `Feature engineering decisions` → `Forecasting models`")
st.caption(f"Source: `{ROOT / 'data' / 'raw' / 'dataset.csv'}` · "
           f"{len(df):,} valid readings · {df.SystemCodeNumber.nunique()} lots · "
           f"{df.date.nunique()} observed dates · {df.timestamp.min():%d %b %Y} to {df.timestamp.max():%d %b %Y}")

tabs = st.tabs([
    "Problem-Driven Research Questions",
    "Hypothesis Testing",
    "Distribution & Data Quality",
    "Correlation & Feature Discovery",
    "EDA Insights",
    "Statistical Decisions Taken",
])


def result_status(result):
    if not result.get("available"):
        st.warning(result.get("reason", "The analysis is unavailable for this dataset."))
        return
    st.write(result["decision"])


def show_test_details(question, null, alternative, test, rationale, interpretation,
                      business, feature_impact):
    st.markdown(f"**Question:** {question}")
    st.markdown(f"**H₀:** {null}  \n**H₁:** {alternative}")
    st.markdown(f"**Test used:** {test}  \n**Why chosen:** {rationale}")
    st.markdown(f"**Interpretation:** {interpretation}")
    st.markdown(f"**Business implication:** {business}")
    st.markdown(f"**Feature engineering impact:** {feature_impact}")


# 1. Research questions
with tabs[0]:
    st.subheader("Why investigate the data before modeling?")
    st.write("The analysis tests whether observed operating conditions carry demand information. "
             "Results inform feature choices and forecasting rationale; associations in this observational dataset do not prove causation.")
    st.markdown(f"**Dataset inspected:** {len(df):,} raw readings across "
                f"{df.SystemCodeNumber.nunique()} lots and {df.date.nunique()} dates. "
                "The source has no price, revenue, or weather columns.")
    questions = [
        ("RQ1 · Traffic congestion", "Does traffic congestion near a lot significantly increase parking occupancy?",
         "A pricing engine may need demand estimates that account for nearby congestion."),
        ("RQ2 · Special days", "Do special days create demand patterns different from normal days?",
         "Special-day demand changes may affect demand-aware pricing and planning."),
        ("RQ3 · Queue pressure", "Does queue length act as an observable indicator of parking demand pressure?",
         "Queue buildup may provide an operational signal for near-term occupancy."),
        ("RQ4 · Vehicle types", "Do readings associated with different vehicle types show different occupancy behavior?",
         "A measurable relationship could motivate evaluating vehicle-type features."),
        ("RQ5 · Temporal structure", "Does occupancy vary systematically by hour and day?",
         "Repeatable patterns support calendar and lag-based forecasting features."),
    ]
    for title, question, relevance in questions:
        with st.container(border=True):
            st.markdown(f"#### {title}")
            st.write(question)
            st.caption(f"Business relevance: {relevance}")
    st.info("Weather and customer prices are not recorded in the supplied CSV. This page does not estimate weather effects or claim a causal price response.")


# 2. Hypothesis tests
with tabs[1]:
    st.subheader("Hypothesis tests on observed readings")
    st.caption("Significance threshold α = 0.05. Tests are observational; repeated readings may be correlated within lot and date.")

    with st.container(border=True):
        st.markdown("#### Hypothesis 1 · Traffic condition and occupancy")
        show_test_details(
            "Does occupancy differ across low, average, and high nearby traffic conditions?",
            "Mean occupancy rate is equal across traffic categories.",
            "At least one traffic category has a different mean occupancy rate.",
            "One-way ANOVA", "The outcome is continuous and the explanatory variable has three observed categories.",
            "Traffic group means are compared below. ANOVA establishes a group difference, if significant; it does not alone establish that increasing traffic causes occupancy to rise.",
            "If the observed differences are reliable, traffic context can improve demand estimates used by operators.",
            "The existing pipeline maps TrafficConditionNearby to the ordinal `traffic` feature (low=0, average=1, high=2). The test evaluates whether retaining that signal is empirically reasonable.")
        if traffic_result.get("available"):
            c1, c2 = st.columns(2)
            c1.metric("F statistic", f"{traffic_result['statistic']:.3f}")
            c2.metric("p-value", f"{traffic_result['p_value']:.4g}")
            traffic_table = pd.DataFrame(traffic_result["groups"]).T
            traffic_table.index.name = "Traffic condition"
            traffic_table["mean"] = traffic_table["mean"].map(lambda x: f"{x:.1%}")
            traffic_table["median"] = traffic_table["median"].map(lambda x: f"{x:.1%}")
            traffic_table["std"] = traffic_table["std"].map(lambda x: f"{x:.1%}")
            st.dataframe(traffic_table, use_container_width=True)
            st.caption(f"η² = {traffic_result['eta_squared']:.4f} (share of observed occupancy variation associated with group membership).")
            result_status(traffic_result)
        else:
            result_status(traffic_result)

    with st.container(border=True):
        st.markdown("#### Hypothesis 2 · Special-day effect")
        show_test_details(
            "Does mean occupancy differ between special and regular dates?",
            "Daily mean occupancy is equal on special and regular dates.",
            "Daily mean occupancy differs on special and regular dates.",
            "Welch independent-samples t-test on daily mean occupancy",
            "The test compares two groups and does not assume equal variance. Daily means avoid treating every reading from one date as independent.",
            "The confidence interval is for (special-day mean − regular-day mean). A p-value below 0.05 indicates evidence of a difference in this dataset.",
            "A consistent shift can be useful for event-day staffing and demand planning; it does not guarantee a price response.",
            "`IsSpecialDay` is already included in the forecasting feature set. This comparison provides an empirical check on that design choice.")
        if special_result.get("available"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Regular-day mean", f"{special_result['regular_mean']:.1%}", f"n={special_result['regular_n_days']} dates")
            c2.metric("Special-day mean", f"{special_result['special_mean']:.1%}", f"n={special_result['special_n_days']} dates")
            c3.metric("Difference", f"{special_result['difference']:+.1%}", "special − regular")
            st.write(f"t = {special_result['statistic']:.3f} · p = {special_result['p_value']:.4g} · "
                     f"95% CI for difference: [{special_result['ci_low']:+.1%}, {special_result['ci_high']:+.1%}]")
            result_status(special_result)
        else:
            result_status(special_result)

    with st.container(border=True):
        st.markdown("#### Hypothesis 3 · Queue pressure and occupancy")
        show_test_details(
            "Is queue length associated with occupancy rate, and does it increase as occupancy increases?",
            "Queue length has no linear association with occupancy rate; the regression slope is zero.",
            "Queue length has a non-zero linear association with occupancy rate, with a positive slope expected.",
            "Pearson correlation significance test and simple linear regression",
            "Both variables are numeric. Pearson tests linear association; regression estimates the occupancy-rate change per additional queued vehicle.",
            "Correlation and slope statistics are shown below. A significant association supports predictive usefulness, not a causal claim.",
            "Queue buildup can serve as an observable pressure signal for monitoring and short-term demand estimation.",
            "`QueueLength` is already an input to the forecast feature set. A positive, significant relationship supports retaining it and evaluating its incremental value.")
        if queue_result.get("available"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Pearson r", f"{queue_result['r']:.3f}")
            c2.metric("Correlation p-value", f"{queue_result['correlation_p']:.4g}")
            c3.metric("Regression slope", f"{queue_result['slope']:.5f}")
            st.write(f"n = {queue_result['n']:,} · regression p = {queue_result['regression_p']:.4g} · "
                     f"R² = {queue_result['r_squared']:.3f} · slope 95% CI "
                     f"[{queue_result['slope_ci_low']:.5f}, {queue_result['slope_ci_high']:.5f}]")
            result_status(queue_result)
        else:
            result_status(queue_result)

    with st.container(border=True):
        st.markdown("#### Hypothesis 4 · Capacity utilization")
        st.markdown("**Question:** How often do observed lots exceed 80% of their stated capacity?  \n"
                    "**Measure:** OccupancyRate = Occupancy / Capacity. This is a descriptive saturation analysis, not a test that capacity is exceeded.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Readings above 80%", f"{utilization_result['above_count']:,} / {utilization_result['n']:,}")
        c2.metric("Share above 80%", f"{utilization_result['above_share']:.1%}")
        c3.metric("Mean occupancy rate", f"{utilization_result['mean']:.1%}")
        st.write(f"Median occupancy rate: {utilization_result['median']:.1%}. "
                 "Frequent high utilization means less observed spare capacity, which can make demand-aware allocation worth evaluating.")
        lot_util = pd.DataFrame(utilization_result["per_lot"])
        lot_util["mean"] = lot_util["mean"].map(lambda x: f"{x:.1%}")
        lot_util["share_above_threshold"] = lot_util["share_above_threshold"].map(lambda x: f"{x:.1%}")
        st.dataframe(lot_util.rename(columns={"SystemCodeNumber": "Lot", "readings": "Readings",
                                              "mean": "Mean occupancy rate", "share_above_threshold": "Readings >80%"}),
                     hide_index=True, use_container_width=True)
        st.caption("Business implication: frequent near-capacity readings support testing demand-sensitive pricing, subject to actual price-response evidence.")

    with st.container(border=True):
        st.markdown("#### Hypothesis 5 · Temporal dependency")
        show_test_details(
            "Does occupancy vary by hour of day?",
            "Mean occupancy is equal across observed hours.",
            "At least one observed hour has a different mean occupancy.",
            "One-way ANOVA across hour-of-day groups, using daily park-wide means as test observations",
            "Occupancy is numeric and hour is a categorical time group. Daily aggregation reduces within-date pseudo-replication; the displayed profile summarizes all raw readings by hour.",
            "The F-test and mean occupancy by hour are shown below. The data covers daytime operating readings, so it cannot establish overnight patterns.",
            "Predictable time-of-day and day-of-week structure can help anticipate demand before a lot fills.",
            "The existing forecaster uses `hour`, `dow`, occupancy lags, and same-time-yesterday occupancy. Temporal structure supports those features and the forecasting task.")
        if temporal_result.get("available"):
            st.write(f"F = {temporal_result['statistic']:.3f} · p = {temporal_result['p_value']:.4g}")
            hourly = temporal_result["hourly"].copy()
            hourly["mean"] = hourly["mean"].map(lambda x: f"{x:.1%}")
            hourly["std"] = hourly["std"].map(lambda x: f"{x:.1%}")
            st.dataframe(hourly.rename(columns={"hour": "Hour", "count": "Readings", "mean": "Mean occupancy", "std": "Std. dev."}),
                         hide_index=True, use_container_width=True)
            result_status(temporal_result)
        else:
            result_status(temporal_result)

    with st.container(border=True):
        st.markdown("#### Vehicle-type research question")
        st.markdown("**Question:** Do recorded occupancy rates differ across vehicle-type labels?  \n"
                    "**Test:** One-way ANOVA across the observed vehicle categories.")
        if vehicle_result.get("available"):
            c1, c2 = st.columns(2)
            c1.metric("F statistic", f"{vehicle_result['statistic']:.3f}")
            c2.metric("p-value", f"{vehicle_result['p_value']:.4g}")
            vtable = pd.DataFrame(vehicle_result["groups"]).T
            vtable.index.name = "Vehicle type"
            vtable["mean"] = vtable["mean"].map(lambda x: f"{x:.1%}")
            vtable["median"] = vtable["median"].map(lambda x: f"{x:.1%}")
            st.dataframe(vtable, use_container_width=True)
            result_status(vehicle_result)
        else:
            result_status(vehicle_result)
        st.caption("VehicleType is present in the source data but is not currently an input in the forecasting feature list. Group differences would justify validation as a candidate, not automatic inclusion.")


# 3. Distribution and data quality
with tabs[2]:
    st.subheader("Distribution and data quality")
    st.write("These summaries describe the supplied observations before drawing modeling conclusions. Outliers are flagged for review, not automatically removed.")
    metric_rows = []
    for column in ("Occupancy", "OccupancyRate", "QueueLength"):
        values = distribution_result[column]
        metric_rows.append({"Measure": column, "n": values["n"], "Mean": values["mean"],
                            "Median": values["median"], "Variance": values["variance"],
                            "Std. deviation": values["std"], "Skewness": values["skewness"],
                            "Kurtosis (excess)": values["kurtosis"]})
    table = pd.DataFrame(metric_rows).set_index("Measure")
    st.dataframe(table.style.format({col: "{:.4f}" for col in table.columns if col != "n"}
                                    | {"n": "{:,.0f}"}), use_container_width=True)
    st.caption("Kurtosis is Fisher excess kurtosis (normal distribution = 0). Occupancy rate is clipped to [0, 1] for the analysis view.")

    sns.set_theme(style="whitegrid")
    occ_fig, occ_ax = plt.subplots(figsize=(8, 3.5))
    sns.histplot(df["Occupancy"].dropna(), bins=35, kde=True, stat="density", ax=occ_ax, color="#4267ac")
    occ_ax.set(title="Observed occupancy distribution", xlabel="Occupancy (vehicles)", ylabel="Density")
    st.pyplot(occ_fig, clear_figure=True)

    rate_fig, rate_ax = plt.subplots(figsize=(8, 3.5))
    sns.histplot(df["OccupancyRate"].dropna(), bins=35, kde=True, stat="density", ax=rate_ax, color="#258277")
    rate_ax.set(title="Occupancy rate distribution", xlabel="Occupancy / capacity", ylabel="Density")
    st.pyplot(rate_fig, clear_figure=True)

    queue_fig, queue_ax = plt.subplots(figsize=(8, 3.5))
    sns.histplot(df["QueueLength"].dropna(), discrete=True, kde=False, ax=queue_ax, color="#bb7737")
    queue_ax.set(title="Queue-length distribution", xlabel="Queue length", ylabel="Readings")
    st.pyplot(queue_fig, clear_figure=True)

    b1, b2, b3 = st.columns(3)
    for column, title, container in (("Occupancy", "Occupancy boxplot", b1),
                                     ("OccupancyRate", "Occupancy-rate boxplot", b2),
                                     ("QueueLength", "Queue-length boxplot", b3)):
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.boxplot(y=df[column], ax=ax, color="#8ca6c9")
        ax.set_title(title)
        container.pyplot(fig, clear_figure=True)

    st.markdown("#### Normality investigation · Shapiro–Wilk")
    normality = distribution_result["occupancy_normality"]
    st.write(f"W = {normality['statistic']:.5f} · p = {normality['p_value']:.4g} · "
             f"n tested = {normality['n_tested']:,} (deterministic sample cap of 5,000 when needed).")
    if normality["p_value"] < .05:
        st.info("At α = 0.05, the normality null is rejected for occupancy. Large samples can detect small deviations; inspect distribution shape and use robust summaries alongside parametric tests.")
    else:
        st.info("At α = 0.05, the test does not provide evidence against normality for the tested sample. This is not proof that occupancy is normally distributed.")

    st.markdown("#### Outlier investigation · 1.5 × IQR fences")
    iqr_rows = []
    for column, values in distribution_result["iqr_outliers"].items():
        iqr_rows.append({"Measure": column, "Lower fence": values["lower_fence"],
                         "Upper fence": values["upper_fence"], "Outlier count": values["count"],
                         "Outlier percent": values["percent"]})
    st.dataframe(pd.DataFrame(iqr_rows).set_index("Measure").style.format(
        {"Lower fence": "{:.2f}", "Upper fence": "{:.2f}", "Outlier percent": "{:.2f}%"}), use_container_width=True)
    st.info("IQR flags identify unusually low or high observations, not data errors. In parking operations, high occupancy and queue outliers may be real congestion events. Keep them unless source validation identifies a recording problem; tree models are generally less sensitive to extreme target values than methods assuming Gaussian noise.")


# 4. Correlation and feature discovery
with tabs[3]:
    st.subheader("Correlation and feature discovery")
    st.caption("Pearson correlations describe linear association. Encoded traffic is ordinal by design; nominal labels such as vehicle type are analyzed as groups instead.")
    corr = correlation_result["matrix"]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, cmap="vlag", center=0, vmin=-1, vmax=1, annot=True, fmt=".2f",
                square=True, linewidths=.4, ax=ax)
    ax.set_title("Correlation heatmap · observed numeric variables")
    st.pyplot(fig, clear_figure=True)
    pos_col, neg_col = st.columns(2)
    with pos_col:
        st.markdown("**Top positive correlations**")
        st.dataframe(pd.DataFrame(correlation_result["positive"], columns=["Variable 1", "Variable 2", "r"])
                     .style.format({"r": "{:.3f}"}), hide_index=True, use_container_width=True)
    with neg_col:
        st.markdown("**Top negative correlations**")
        st.dataframe(pd.DataFrame(correlation_result["negative"], columns=["Variable 1", "Variable 2", "r"])
                     .style.format({"r": "{:.3f}"}), hide_index=True, use_container_width=True)
    st.info("High correlation can reflect shared definitions (OccupancyRate uses Occupancy and Capacity) or shared time/lot structure. It is not a causal effect and should not be used alone to remove a feature.")

    st.markdown("#### Feature discovery")
    traffic_evidence = (f"ANOVA p={traffic_result['p_value']:.3g}; "
                       + ", ".join(f"{k} mean={v['mean']:.1%}" for k, v in traffic_result.get("groups", {}).items())) if traffic_result.get("available") else "Traffic test unavailable."
    special_evidence = (f"Welch t-test p={special_result['p_value']:.3g}; difference="
                        f"{special_result['difference']:+.1%} (special − regular).") if special_result.get("available") else "Special-day test unavailable."
    queue_evidence = (f"Pearson r={queue_result['r']:.3f}, p={queue_result['correlation_p']:.3g}; "
                      f"slope={queue_result['slope']:.4f}.") if queue_result.get("available") else "Queue association test unavailable."
    temporal_evidence = (f"Hour ANOVA p={temporal_result['p_value']:.3g}; "
                         f"{temporal_result['hourly']['hour'].nunique()} observed hours.") if temporal_result.get("available") else "Temporal test unavailable."
    vehicle_evidence = (f"Vehicle-type ANOVA p={vehicle_result['p_value']:.3g}; "
                        f"{len(vehicle_result['groups'])} categories.") if vehicle_result.get("available") else "Vehicle-type test unavailable."
    candidates = [
        ("QueueLength", queue_evidence, "Captures a direct operational pressure signal; already included in the model inputs."),
        ("TrafficConditionNearby → traffic", traffic_evidence, "The existing ordinal encoding provides context that may help occupancy forecasts."),
        ("IsSpecialDay", special_evidence, "Represents calendar-specific demand shifts; already included in the model inputs."),
        ("Hour and day-of-week", temporal_evidence, "Represent recurring time structure; hour, day-of-week, and lag features are already included."),
        ("OccupancyRate", f"Mean={utilization_result['mean']:.1%}; {utilization_result['above_share']:.1%} of records exceed 80%.", "Normalizes occupancy by lot size and serves as the forecast target and current-state signal."),
        ("VehicleType", vehicle_evidence, "A potential candidate if useful out-of-sample; it is not in the current forecast feature list."),
    ]
    for name, evidence, benefit in candidates:
        with st.container(border=True):
            st.markdown(f"**{name}**")
            st.write(f"Observed relationship / statistical evidence: {evidence}")
            st.write(f"Potential modeling benefit: {benefit}")


# 5. EDA insights and raw-data views
with tabs[4]:
    st.subheader("Exploratory analysis of observed operating patterns")
    st.caption("All charts below are derived from the supplied CSV. Group differences are descriptive and do not establish causality.")

    c1, c2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="TrafficConditionNearby", y="OccupancyRate",
                order=[x for x in ["low", "average", "high"] if x in df.TrafficConditionNearby.unique()], ax=ax)
    ax.set(title="Occupancy rate by traffic condition", xlabel="Nearby traffic", ylabel="Occupancy rate")
    c1.pyplot(fig, clear_figure=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=df, x="VehicleType", y="OccupancyRate", ax=ax)
    ax.set(title="Occupancy rate by vehicle type", xlabel="Recorded vehicle type", ylabel="Occupancy rate")
    c2.pyplot(fig, clear_figure=True)

    c1, c2 = st.columns(2)
    special_view = df.assign(DayType=np.where(df.IsSpecialDay == 1, "Special day", "Regular day"))
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(data=special_view, x="DayType", y="OccupancyRate", ax=ax, color="#81a9c8")
    ax.set(title="Occupancy rate by special-day flag", xlabel="Date type", ylabel="Occupancy rate")
    c1.pyplot(fig, clear_figure=True)

    daily = df.groupby("date").agg(occupancy_rate=("OccupancyRate", "mean"),
                                    occupancy=("Occupancy", "mean")).reset_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.lineplot(data=daily, x="date", y="occupancy_rate", marker="o", ax=ax, color="#4267ac")
    ax.set(title="Mean occupancy rate over observed dates", xlabel="Date", ylabel="Mean occupancy rate")
    ax.tick_params(axis="x", rotation=35)
    c2.pyplot(fig, clear_figure=True)

    c1, c2 = st.columns(2)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.regplot(data=df, x="QueueLength", y="Occupancy", ax=ax,
                scatter_kws={"alpha": .16, "s": 12}, line_kws={"color": "#b33b3b"}, ci=None)
    ax.set(title="Queue length vs observed occupancy", xlabel="Queue length", ylabel="Occupancy (vehicles)")
    c1.pyplot(fig, clear_figure=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.scatterplot(data=df, x="Capacity", y="Occupancy", hue="SystemCodeNumber",
                    alpha=.45, s=18, legend=False, ax=ax)
    ax.set(title="Capacity vs observed occupancy", xlabel="Lot capacity", ylabel="Occupancy (vehicles)")
    c2.pyplot(fig, clear_figure=True)

    st.markdown("#### Data-driven insight cards")
    if weekend_result.get("available"):
        weekend_direction = "higher" if weekend_result["difference"] > 0 else "lower"
        insight_cards.insert(2, {
            "title": "Weekday and weekend pattern",
            "observation": f"Daily mean occupancy is {weekend_direction} on weekends by {abs(weekend_result['difference']):.1%}.",
            "evidence": f"Weekday {weekend_result['weekday_mean']:.1%} (n={weekend_result['weekday_n_days']} dates) vs weekend {weekend_result['weekend_mean']:.1%} (n={weekend_result['weekend_n_days']} dates); Welch t-test p={weekend_result['p_value']:.3g}.",
            "business_meaning": "A weekday/weekend difference may inform staffing and demand expectations; assess its predictive value on held-out dates before changing operations."
        })
    for card in insight_cards:
        with st.container(border=True):
            st.markdown(f"##### {card['title']}")
            st.markdown(f"**Observation:** {card['observation']}")
            st.markdown(f"**Evidence:** {card['evidence']}")
            st.markdown(f"**Business meaning:** {card['business_meaning']}")

    v_means = df.groupby("VehicleType").OccupancyRate.mean().sort_values(ascending=False)
    st.markdown(f"**Vehicle type:** the highest observed group mean is `{v_means.index[0]}` at "
                f"{v_means.iloc[0]:.1%}; see the group test in Hypothesis Testing. The source does not say that these labels are vehicle-specific occupancy counts.")
    if weekend_result.get("available"):
        st.caption(f"Supplementary weekday/weekend comparison: daily-mean Welch test p={weekend_result['p_value']:.4g}; "
                   f"95% CI for weekend − weekday = [{weekend_result['ci_low']:+.1%}, {weekend_result['ci_high']:+.1%}]. "
                   f"{weekend_result['decision']}")


# 6. Statistical decisions
with tabs[5]:
    st.subheader("How evidence connects to the current data-science design")
    st.write("This table documents observed evidence alongside the existing model pipeline. It does not change feature selection, retrain models, or alter pricing.")
    rows = []
    if traffic_result.get("available"):
        rows.append({"Finding": "Traffic condition is associated with occupancy" if traffic_result["p_value"] < .05 else "Traffic effect not established at α=.05",
                     "Evidence": f"One-way ANOVA p={traffic_result['p_value']:.4g}; η²={traffic_result['eta_squared']:.3f}.",
                     "Project decision": "Existing `traffic` encoding (0/1/2) is retained as a forecast input; the association is not interpreted causally."})
    if special_result.get("available"):
        rows.append({"Finding": "Special-day occupancy differs" if special_result["p_value"] < .05 else "Special-day difference not established at α=.05",
                     "Evidence": f"Daily means: regular {special_result['regular_mean']:.1%}, special {special_result['special_mean']:.1%}; Welch p={special_result['p_value']:.4g}; 95% CI [{special_result['ci_low']:.1%}, {special_result['ci_high']:.1%}].",
                     "Project decision": "`IsSpecialDay` remains in the existing forecast feature list; this test documents its observed relevance."})
    if queue_result.get("available"):
        rows.append({"Finding": "Queue length has a positive occupancy association" if queue_result["r"] > 0 and queue_result["correlation_p"] < .05 else "Queue association is weak or not established",
                     "Evidence": f"Pearson r={queue_result['r']:.3f}, p={queue_result['correlation_p']:.4g}; regression slope={queue_result['slope']:.5f}.",
                     "Project decision": "`QueueLength` is already included as a forecast input; its incremental predictive value should be judged with held-out model evaluation."})
    if temporal_result.get("available"):
        rows.append({"Finding": "Occupancy varies by hour" if temporal_result["p_value"] < .05 else "Hourly variation not established at α=.05",
                     "Evidence": f"One-way ANOVA p={temporal_result['p_value']:.4g}; {temporal_result['hourly']['hour'].nunique()} hours observed.",
                     "Project decision": "Calendar fields (`hour`, `dow`) and occupancy lags are included in the existing forecaster; the dataset contains daytime readings only."})
    if vehicle_result.get("available"):
        rows.append({"Finding": "Vehicle-type group means differ" if vehicle_result["p_value"] < .05 else "Vehicle-type difference not established at α=.05",
                     "Evidence": f"One-way ANOVA p={vehicle_result['p_value']:.4g}; {len(vehicle_result['groups'])} recorded labels.",
                     "Project decision": "`VehicleType` is not currently a forecast input. Treat it as a candidate for a future leakage-safe, out-of-sample comparison, not as an automatically accepted feature."})
    rows.append({"Finding": "Lot size affects raw occupancy scale",
                 "Evidence": f"Lot capacities range from {df.Capacity.min():.0f} to {df.Capacity.max():.0f}; the pipeline defines OccupancyRate = Occupancy / Capacity.",
                 "Project decision": "Use normalized occupancy rate for the forecasting target and compare lots on a common 0–1 scale."})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
    st.markdown("#### Interpretation limits")
    st.markdown("- The source dataset has no weather, transaction price, or realized revenue fields. Weather and price causality cannot be tested from these readings.\n"
                "- The analyses use observational readings. Repeated measurements within lots and dates can violate independent-observation assumptions; p-values should be read as exploratory evidence.\n"
                "- Statistical association is not the same as forecasting usefulness. The project's chronological held-out evaluation remains the appropriate check for predictive value.\n"
                "- VehicleType labels describe the recorded rows; the dataset does not document that occupancy is split into vehicle-specific counts.")
