"""Traceable documentation of the current ParkedIn feature pipeline."""
import pandas as pd
import streamlit as st

from _setup import ROOT  # noqa: F401  # adds project root to sys.path
from agents.data_agent import FEATURES
from analytics.statistics import load_dataset

st.set_page_config(page_title="ParkedIn: Feature Engineering", layout="wide")
st.title("Feature Engineering")
st.caption("Trace how source readings become the inputs used by the existing occupancy forecasters.")


@st.cache_data(show_spinner="Loading source and engineered datasets…")
def load_feature_data():
    return load_dataset("raw"), load_dataset("processed")


try:
    raw, processed = load_feature_data()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Project dataset unavailable: {exc}")
    st.stop()

st.markdown("`Original parking readings` → `timestamp alignment and cleaning` → `engineered features` → `XGBoost occupancy forecasts`")
st.caption(f"Actual project files · raw rows: {len(raw):,} · processed lot-slot rows: {len(processed):,} · "
           f"lots: {processed.SystemCodeNumber.nunique()} · model input fields: {len(FEATURES)}")

source_groups = {
    "Identifiers": ["ID", "SystemCodeNumber"],
    "Lot attributes": ["Capacity", "Latitude", "Longitude"],
    "Observed parking and context": ["Occupancy", "VehicleType", "TrafficConditionNearby",
                                      "QueueLength", "IsSpecialDay"],
    "Time fields": ["LastUpdatedDate", "LastUpdatedTime"],
}

st.subheader("Original features")
st.write("Fields below come directly from the actual source CSV. Weekend, rush-hour, and season-category flags are not explicit source columns.")
for group, columns in source_groups.items():
    present = [column for column in columns if column in raw.columns]
    st.markdown(f"**{group}:** " + (", ".join(f"`{column}`" for column in present) if present else "None"))

st.subheader("Processing before feature creation")
preparation = [
    ("Timestamp alignment", "Parse LastUpdatedDate + LastUpdatedTime, round to the nearest 30 minutes, and keep the latest reading for each lot and slot."),
    ("Common lot-slot grid", "Align lots to shared slots and skip rare slots with readings from fewer than 80% of lots."),
    ("Missing values", "Interpolate Occupancy, QueueLength, and encoded traffic within each lot in both directions. Capacity and coordinates use the first value for that lot."),
    ("Target and split", "Calculate bounded occupancy rate, then split whole dates chronologically into train, validation, and test sets."),
]
for i, (title, detail) in enumerate(preparation, 1):
    with st.container(border=True):
        st.markdown(f"**{i}. {title}**")
        st.write(detail)

specs = [
    ("occ_rate", "Occupancy, Capacity", "clip(Occupancy / Capacity, 0, 1).", "Puts different-sized lots on a common utilization scale.", "Normalizes the outcome and current state, reducing scale differences caused by capacity.", "Operators can compare utilization across lots as a percentage.", "Forecast target at future horizons and current-state model input."),
    ("occ_lag_1", "occ_rate within SystemCodeNumber", "Previous row's occupancy rate for the same lot (group shift 1).", "Captures the latest observed demand state.", "Represents short-lag serial dependence.", "Recent fullness is a starting point for near-term decisions.", "Provides the model with the most recent occupancy level."),
    ("occ_lag_2", "occ_rate within SystemCodeNumber", "Occupancy rate two rows earlier for the same lot.", "Adds a slightly longer recent history.", "Lets the model distinguish a brief fluctuation from developing demand.", "Shows whether demand was already elevated before the current reading.", "Allows nonlinear tree splits on another past state."),
    ("occ_lag_3", "occ_rate within SystemCodeNumber", "Occupancy rate three rows earlier for the same lot.", "Extends recent history to three rows on the aligned slot grid.", "Captures dependence over a longer short-term lag.", "Adds context on how demand was evolving.", "Lets trees use several recent states without imposing a linear form."),
    ("occ_roll_3", "Prior occ_rate values within SystemCodeNumber", "Mean of the prior three rows: shift(1), then rolling(3).", "Smooths one-slot noise without including the current value in its own average.", "A local moving average summarizes recent underlying demand.", "Can reduce reactions to one isolated fluctuation.", "Provides a compact summary of recent occupancy history."),
    ("occ_diff", "occ_rate within SystemCodeNumber", "Current occupancy rate minus the previous row (group diff).", "Encodes whether occupancy is rising or falling.", "Represents the first difference and local direction of change.", "A rising lot may need attention before it becomes full.", "Adds direction information alongside occupancy levels."),
    ("occ_same_slot_yesterday", "occ_rate grouped by lot and clock time", "Previous observed row for the same lot and clock time; usually the prior date's matching time.", "Provides a same-time historical reference.", "Captures repeated within-day temporal structure.", "Compares current demand with the lot's prior matching-clock-time reading.", "Adds a calendar-aligned lag distinct from recent slots."),
    ("hour", "Aligned slot timestamp", "Hour + minute / 60, as a decimal hour.", "Represents intraday position.", "Allows the model to learn observed time-of-day effects.", "Parking use may differ across operating hours.", "Continuous calendar input; a separate rush-hour flag is not created."),
    ("dow", "Aligned slot timestamp", "Day of week as an integer: Monday=0 through Sunday=6.", "Represents weekly calendar position.", "Allows learning weekday and day-specific patterns.", "Demand may differ between working days and weekends.", "Numeric calendar input; no separate weekend flag is engineered."),
    ("traffic", "TrafficConditionNearby", "Map low→0, average→1, high→2; interpolate missing values by lot and round.", "Converts ordered categories to a numeric model input.", "Preserves category ordering in a compact encoding.", "Nearby congestion may coincide with changing parking demand.", "Adds traffic context; assumes equal numeric steps between levels."),
    ("QueueLength", "QueueLength", "Interpolate missing values by lot and round to an integer.", "Retains an operational measure of waiting vehicles.", "Provides a measurable demand-pressure variable.", "Queue buildup can signal constrained availability.", "Adds pressure information beyond occupancy history."),
    ("IsSpecialDay", "Source IsSpecialDay grouped by date", "Take the daily maximum, fill missing date flags with 0, and cast to integer.", "Makes one consistent special-day flag across each date.", "Represents calendar events that may shift demand.", "Special dates can change staffing and demand expectations.", "Binary calendar input; source does not specify holiday types."),
    ("Capacity", "Capacity grouped by SystemCodeNumber", "Use the first recorded capacity for each lot and repeat it on that lot's rows.", "Retains the size of each parking facility.", "Adds facility scale after occupancy normalization.", "Capacity is essential context for interpreting lot utilization.", "Provides the forecaster with lot-size context."),
]

st.subheader("Engineered features used by the forecasting models")
st.write("Feature names and order come from `agents/data_agent.py`. Each lineage row maps raw sources to its creation rule; expand a feature for the full rationale.")
lineage = pd.DataFrame([{"Original feature(s)": s, "Engineered feature": name, "Creation logic": logic}
                         for name, s, logic, *_ in specs])
st.dataframe(lineage, hide_index=True, use_container_width=True)

for name, source, logic, why, stats_motivation, business, ml_impact in specs:
    with st.expander(name):
        c1, c2 = st.columns(2)
        c1.markdown(f"**Feature name**  \n`{name}`")
        c2.markdown(f"**Original feature(s)**  \n{source}")
        st.markdown(f"**Creation logic**  \n{logic}")
        c1.markdown(f"**Why created**  \n{why}")
        c2.markdown(f"**Statistical motivation**  \n{stats_motivation}")
        c1.markdown(f"**Business motivation**  \n{business}")
        c2.markdown(f"**Expected ML impact**  \n{ml_impact}")

st.subheader("Model input order")
st.code(" → ".join(FEATURES), language="text")
missing = sorted(set(FEATURES).difference(processed.columns))
if missing:
    st.error(f"Processed data is missing configured model inputs: {', '.join(missing)}")
else:
    st.success(f"All {len(FEATURES)} configured model inputs are present in the actual processed dataset.")

st.subheader("Feature lineage trace")
st.code(
    "Occupancy + Capacity ──> occ_rate ──> future occupancy target\n"
    "                                      ├──> occ_lag_1 / occ_lag_2 / occ_lag_3\n"
    "                                      ├──> occ_roll_3 and occ_diff\n"
    "                                      └──> occ_same_slot_yesterday\n"
    "LastUpdatedDate + LastUpdatedTime ──> slot ──> hour, dow\n"
    "TrafficConditionNearby ──> ordered traffic code (with within-lot interpolation)\n"
    "QueueLength ──> interpolated, rounded QueueLength\n"
    "IsSpecialDay ──> date-level flag; Capacity ──> lot-level size context",
    language="text")

st.subheader("Other derived pipeline fields")
st.caption("These fields organize and validate the processed dataset; they are not XGBoost inputs unless listed in the model input order above.")
pipeline_fields = [
    ("time", "LastUpdatedDate + LastUpdatedTime", "Parse the source date and time into a datetime.",
     "Provides a consistent timestamp for temporal operations.", "Enables correct chronological ordering.",
     "Makes each record's collection time explicit.", "Supports rounding and sorting; not passed directly to the model."),
    ("slot", "time", "Round the parsed timestamp to the nearest 30 minutes.",
     "Aligns irregular sensor timestamps to the simulation's half-hour unit.", "Creates comparable time bins across lots.",
     "Matches the system's 30-minute pricing/forecast step.", "Groups observations into consistent replay slots; not a model input."),
    ("date", "slot", "Normalize slot timestamp to midnight of its calendar date.",
     "Creates a date key for day-level aggregation and flags.", "Supports date-level summaries and chronological partitioning.",
     "Allows special-day state to be consistent for the day.", "Used for IsSpecialDay propagation and train/validation/test splitting."),
    ("split", "date and configured train/validation fractions", "Assign whole dates in chronological order to train, validation, or test.",
     "Keeps later dates held out for forecasting evaluation.", "Avoids randomly mixing future and past rows across partitions.",
     "Provides a realistic evaluation of future-period forecasting.", "Controls model fitting and evaluation; never passed as a predictor."),
    ("Corrected coordinates", "Latitude, Longitude", "For the identified out-of-region coordinate, replace it with a nearby point based on the other lots' medians.",
     "Prevents one known coordinate error from distorting the map display.", "This is a data-quality correction, not a statistical demand feature.",
     "Keeps all lots visible in the intended map area.", "Coordinates are not included in the current forecasting feature list."),
]
for name, source, logic, why, statistical, business, impact in pipeline_fields:
    with st.expander(f"{name} · data preparation"):
        st.markdown(f"**Original source:** {source}  \n**Creation logic:** {logic}")
        c1, c2 = st.columns(2)
        c1.markdown(f"**Why created:** {why}  \n**Statistical motivation:** {statistical}")
        c2.markdown(f"**Business motivation:** {business}  \n**Expected ML impact:** {impact}")

st.info("Weekend Flag, Rush Hour Flag, and Season Category are not separate engineered columns in the current pipeline. The source-provided IsSpecialDay flag is not a categorized holiday calendar. This page documents existing features only; it does not add model inputs or change training.")
