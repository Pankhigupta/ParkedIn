"""Explain the existing ParkedIn machine-learning and forecasting pipeline."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from _setup import ROOT  # noqa: F401
from agents.data_agent import FEATURES
from agents.forecast_agent import ForecastAgent
from analytics.model_evaluation import clustering_evaluation
from analytics.statistics import load_dataset, load_evaluation_metrics, model_catalog
from common import FORECAST_MODEL_PATH, PATTERN_MODEL_PATH, load_config

st.set_page_config(page_title="ParkedIn: Machine Learning", layout="wide")
st.title("Machine Learning")
st.caption("Inspect the fitted models, feature design, evaluation split, and explanations from saved artifacts.")


@st.cache_data(show_spinner="Loading processed project data…")
def load_processed():
    return load_dataset("processed")


@st.cache_resource(show_spinner="Loading saved forecast models…")
def load_saved_forecaster():
    return ForecastAgent(horizon=6)


try:
    data = load_processed()
except (FileNotFoundError, ValueError) as exc:
    st.error(f"Processed dataset unavailable: {exc}")
    st.stop()

config = load_config()
train_days = sorted(data.loc[data.split == "train", "date"].dropna().unique())
val_days = sorted(data.loc[data.split == "val", "date"].dropna().unique())
test_days = sorted(data.loc[data.split == "test", "date"].dropna().unique())

st.markdown("#### Dataset preparation")
st.write("The existing data agent aligns readings to half-hour slots, keeps the latest lot-slot reading, creates a common lot-slot grid, interpolates selected missing values within each lot, calculates occupancy rate, and creates chronological features.")
st.caption(f"Processed actual data: {len(data):,} lot-slot rows · {data.SystemCodeNumber.nunique()} lots · "
           f"{data.slot.min():%d %b %Y %H:%M}–{data.slot.max():%d %b %Y %H:%M}")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Dataset Preparation", "Feature Set", "Target Variable", "Train-Test Split",
    "Model Training Pipeline", "Feature Importance & Explainability"
])

with tab1:
    st.subheader("Dataset preparation")
    st.markdown("**Raw input:** `data/raw/dataset.csv`  \n**Modeling data:** `data/processed/features.pkl`")
    st.markdown("1. Parse and round timestamps to 30-minute slots.\n"
                "2. Keep the latest reading for each lot-slot and align lots to common timestamps.\n"
                "3. Interpolate occupancy, queue, and ordinal traffic values within lot where the source has gaps.\n"
                "4. Normalize occupancy by capacity and calculate lags, rolling statistics, and calendar fields.\n"
                "5. Assign entire dates chronologically to train, validation, and test partitions.")
    st.info("Processing details mirror `agents/data_agent.py`; this page reads the saved processed dataset and does not rebuild or alter it.")

with tab2:
    st.subheader("Forecast feature set")
    st.write("The forecasting models use the configured features below, in the exact order imported from `agents.data_agent.FEATURES`.")
    feature_notes = {
        "occ_rate": "Current occupancy divided by capacity; also the normalized target scale.",
        "occ_lag_1": "Previous occupancy-rate row for the same lot.",
        "occ_lag_2": "Occupancy-rate row two steps back for the same lot.",
        "occ_lag_3": "Occupancy-rate row three steps back for the same lot.",
        "occ_roll_3": "Mean of the previous three occupancy-rate rows (shifted to exclude current row).",
        "occ_diff": "Current occupancy rate minus the previous row.",
        "occ_same_slot_yesterday": "Previous observed occupancy at the same clock time for that lot.",
        "hour": "Decimal hour derived from aligned timestamp.",
        "dow": "Day-of-week integer (Monday=0 through Sunday=6).",
        "traffic": "Traffic category encoded low=0, average=1, high=2.",
        "QueueLength": "Interpolated and rounded queue length.",
        "IsSpecialDay": "Date-level binary special-day flag from the source.",
        "Capacity": "Lot capacity repeated as lot context.",
    }
    feature_table = pd.DataFrame([{"Feature": name, "Present in processed data": name in data.columns,
                                   "Role": feature_notes.get(name, "Configured model input.")}
                                  for name in FEATURES])
    st.dataframe(feature_table, hide_index=True, use_container_width=True)
    st.markdown("**Scaling:** XGBoost and Random Forest use tree splits and the forecast feature pipeline does not standardize these inputs. The separate K-Means pattern model uses `StandardScaler` over occupancy rate, queue length, and traffic before clustering.")
    st.markdown("**Encoding:** traffic is ordinal-coded; day-of-week is an integer; `IsSpecialDay` is binary. VehicleType, lot IDs, and coordinates are not forecast inputs. The source does not provide separate weekend, rush-hour, or season-category flags.")

with tab3:
    st.subheader("Target variable")
    st.markdown("**Target:** future `occ_rate` (occupancy ÷ capacity, clipped to 0–1).")
    st.write("For each lot and forecast horizon h, the target is the occupancy rate h rows into that lot's chronological series. The system trains six separate XGBoost regressors for h=1…6, corresponding to 30 minutes through three hours ahead.")
    st.caption("The same normalized occupancy scale makes target errors comparable across lots of different capacities.")

with tab4:
    st.subheader("Chronological train / validation / test split")
    c1, c2, c3 = st.columns(3)
    # The data pipeline stores this partition as "val"; keep the UI label descriptive.
    for column, split_name, split_key, dates in (
            (c1, "Train", "train", train_days),
            (c2, "Validation", "val", val_days),
            (c3, "Test", "test", test_days)):
        subset = data[data.split == split_key]
        column.metric(split_name, f"{len(subset):,} rows", f"{len(dates)} dates")
        if dates:
            column.caption(f"{pd.Timestamp(dates[0]):%d %b %Y}–{pd.Timestamp(dates[-1]):%d %b %Y}")
    st.write(f"Configured split fractions: train={config['train_fraction']:.0%}, "
             f"validation={config['val_fraction']:.0%}, test={1-config['train_fraction']-config['val_fraction']:.0%}.")
    st.info("The split is by whole dates in time order. Validation dates choose the XGBoost tree count with early stopping; final saved XGBoost models are then fitted on train + validation. The held-out test dates are used for final comparison and are not used for fitting.")

with tab5:
    st.subheader("Implemented modeling pipeline")
    stages = [
        ("Forecast target", "Create future occupancy-rate targets independently by lot for six horizons."),
        ("Naive baseline", "Use current occupancy rate as the no-change prediction."),
        ("ARIMA", "Evaluate an ARIMA(2,0,1) time-series baseline per lot."),
        ("Random Forest", "Evaluate a 200-tree RandomForestRegressor for each horizon."),
        ("XGBoost", "Fit a separate XGBRegressor per horizon; use validation early stopping to select tree counts, then save models fitted on train + validation."),
        ("Pattern clustering", "Fit four-cluster K-Means on occupancy rate, queue length, and traffic, with StandardScaler."),
        ("Held-out evaluation", "Compare forecasting MAE on the chronological test partition; save clustering inertia and silhouette diagnostics."),
    ]
    for number, (title, detail) in enumerate(stages, 1):
        with st.container(border=True):
            st.markdown(f"**{number}. {title}**")
            st.write(detail)
    st.markdown("#### Available saved artifacts")
    catalog = pd.DataFrame(model_catalog())
    st.dataframe(catalog[["name", "artifact_type", "exists", "description"]],
                 hide_index=True, use_container_width=True)
    cluster = clustering_evaluation()
    silhouette = cluster.get("silhouette_k4")
    if silhouette is not None:
        st.metric("Saved K-Means silhouette (k=4)", f"{float(silhouette):.3f}")
        st.caption("Silhouette summarizes cluster separation; it is not a forecasting accuracy metric.")

with tab6:
    st.subheader("Global feature importance · saved one-step XGBoost")
    if not FORECAST_MODEL_PATH.is_file():
        st.info("No persisted forecast model is available. Run the existing training process to create the artifact.")
    else:
        try:
            xgb_model = load_saved_forecaster().models[1]
            native_importance = pd.Series(xgb_model.feature_importances_, index=FEATURES,
                                          name="Native XGBoost importance").sort_values()
            fig, ax = plt.subplots(figsize=(9, 5.5))
            native_importance.plot.barh(ax=ax, color="#4d83b8")
            ax.set(title="Native tree importance · horizon 1", xlabel="Relative importance", ylabel="Feature")
            fig.tight_layout()
            st.pyplot(fig, clear_figure=True)
            st.caption(f"{native_importance.index[-1]} has the largest influence on the saved one-step forecast model.")
            saved_metrics = load_evaluation_metrics().get("feature_importance", {})
            if saved_metrics:
                st.write("Training-report importance saved during validation-model evaluation:")
                st.dataframe(pd.Series(saved_metrics, name="Saved validation importance")
                             .sort_values(ascending=False).rename_axis("Feature").reset_index(),
                             hide_index=True, use_container_width=True)
        except Exception as exc:
            st.warning(f"Unable to read the saved model's native importance: {exc}")

    st.markdown("#### SHAP explainability")
    st.write("The persisted XGBoost tree model is compatible with TreeSHAP. Global mean absolute SHAP values summarize contributions across real held-out rows; local SHAP values explain one selected lot-slot prediction relative to the explainer baseline.")
    if FORECAST_MODEL_PATH.is_file():
        complete_test = data[(data.split == "test") & data[FEATURES].notna().all(axis=1)].copy()
        if not complete_test.empty:
            complete_test = complete_test.reset_index(drop=True)
            complete_test["_label"] = complete_test.SystemCodeNumber.astype(str) + " · " + complete_test.slot.astype(str)
            chosen_label = st.selectbox("Local test prediction to explain", complete_test._label.tolist(), key="ml_shap_row")
            if st.button("Calculate SHAP explanations", type="primary"):
                try:
                    import shap
                    background = complete_test.sample(n=min(128, len(complete_test)), random_state=42)
                    local_idx = int(complete_test.index[complete_test._label == chosen_label][0])
                    if local_idx not in background.index:
                        background = pd.concat([background, complete_test.loc[[local_idx]]], axis=0)
                    x_values = background[FEATURES]
                    explainer = shap.TreeExplainer(load_saved_forecaster().models[1])
                    shap_values = np.asarray(explainer.shap_values(x_values))
                    if shap_values.ndim == 3:
                        shap_values = shap_values[:, :, 0]
                    local_position = int(np.flatnonzero(background.index.to_numpy() == local_idx)[0])
                    base = np.asarray(explainer.expected_value).reshape(-1)
                    baseline = float(base[0])
                    st.session_state["ml_shap_result"] = {
                        "global": np.mean(np.abs(shap_values), axis=0),
                        "local": shap_values[local_position],
                        "values": x_values.iloc[local_position].to_numpy(dtype=float),
                        "features": list(FEATURES), "baseline": baseline,
                        "label": chosen_label,
                        "prediction": float(load_saved_forecaster().models[1].predict(
                            x_values.iloc[[local_position]])[0]),
                    }
                except ImportError:
                    st.warning("SHAP is not installed in this environment. Install project dependencies (`pip install -r requirements.txt`) to enable TreeSHAP.")
                except Exception as exc:
                    st.warning(f"SHAP could not explain this saved model in the current environment: {exc}")

            shap_result = st.session_state.get("ml_shap_result")
            if shap_result:
                global_values = pd.Series(shap_result["global"], index=shap_result["features"], name="Mean |SHAP|").sort_values()
                fig, ax = plt.subplots(figsize=(9, 5))
                global_values.plot.barh(ax=ax, color="#478b72")
                ax.set(title="Global SHAP importance · held-out rows", xlabel="Mean absolute SHAP value", ylabel="Feature")
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
                global_top = global_values.index[-1]
                st.caption(f"Across the sampled test rows, {global_top} has the largest average effect on the model's predictions.")

                local = pd.DataFrame({"feature": shap_result["features"],
                                      "contribution": shap_result["local"],
                                      "feature_value": shap_result["values"]})
                local = local.reindex(local.contribution.abs().sort_values(ascending=False).index).head(10)
                local = local.sort_values("contribution")
                fig, ax = plt.subplots(figsize=(9, 5))
                colors = ["#c45b51" if value < 0 else "#39856f" for value in local.contribution]
                ax.barh(local.feature, local.contribution, color=colors)
                ax.axvline(0, color="#333333", linewidth=.8)
                ax.set(title=f"Local SHAP contributions · {shap_result['label']}",
                       xlabel="Contribution to predicted occupancy rate", ylabel="Feature")
                fig.tight_layout()
                st.pyplot(fig, clear_figure=True)
                strongest = local.iloc[local.contribution.abs().argmax()]
                effect = "raises" if strongest.contribution > 0 else "lowers"
                st.caption(f"For this lot and time, {strongest.feature} most strongly {effect} the predicted occupancy; the forecast is {shap_result['prediction']:.0%}.")
                st.dataframe(local.rename(columns={"feature": "Feature", "contribution": "SHAP contribution",
                                                   "feature_value": "Observed feature value"}),
                             hide_index=True, use_container_width=True)
        else:
            st.info("No complete test rows are available for SHAP explanations.")
