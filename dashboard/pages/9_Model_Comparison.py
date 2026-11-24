"""Leaderboard and metric-based comparison of existing ParkedIn models."""
import matplotlib.pyplot as plt
import importlib
import numpy as np
import pandas as pd
import streamlit as st

from _setup import ROOT  # noqa: F401
from analytics import model_evaluation
# Streamlit can retain imported Python modules across page reruns. Reload this
# small, read-only analytics module so newly added evaluation helpers are visible.
model_evaluation = importlib.reload(model_evaluation)
forecast_comparison = model_evaluation.forecast_comparison
implemented_model_catalog = model_evaluation.implemented_model_catalog
frozen_forecast_metrics = getattr(model_evaluation, "frozen_forecast_metrics", None)
from analytics.statistics import load_evaluation_metrics
from common import METRICS_PATH

st.set_page_config(page_title="ParkedIn: Model Comparison", layout="wide")
st.title("Model Comparison")
st.caption("Compare saved chronological test results and inspect metrics recomputed from persisted models, without retraining.")


@st.cache_data(show_spinner="Reading saved evaluation results…")
def load_saved_mae():
    return forecast_comparison()


@st.cache_data(show_spinner="Scoring frozen models on the held-out test split…")
def load_frozen_metrics():
    if frozen_forecast_metrics is None:
        return pd.DataFrame()
    return frozen_forecast_metrics()


saved_mae = load_saved_mae()
if saved_mae.empty:
    st.error(f"No saved forecast metrics found at {METRICS_PATH}. Run the existing training workflow first.")
    st.stop()

st.subheader("Implemented model inventory")
catalog = pd.DataFrame(implemented_model_catalog())
st.dataframe(catalog, hide_index=True, use_container_width=True)
st.caption("Random Forest and ARIMA were evaluated by the training script but their fitted objects/prediction arrays were not persisted. K-Means is a clustering model, so its silhouette score is reported separately rather than ranked by forecast error.")

st.subheader("Saved forecasting leaderboard · test MAE")
st.caption("Canonical leaderboard uses MAE written by the existing training evaluation. Lower occupancy-rate MAE is better.")
long_mae = (saved_mae.rename_axis("horizon")
            .reset_index().melt(id_vars="horizon", var_name="model", value_name="MAE"))
long_mae["Rank"] = long_mae.groupby("horizon")["MAE"].rank(method="min", ascending=True).astype(int)
long_mae = long_mae.sort_values(["horizon", "Rank", "model"])
st.dataframe(long_mae.rename(columns={"horizon": "Slots ahead", "model": "Model", "MAE": "Test MAE"})
             .style.format({"Test MAE": "{:.5f}"}), hide_index=True, use_container_width=True)

fig, ax = plt.subplots(figsize=(10, 5))
for model in saved_mae.columns:
    ax.plot(saved_mae.index, saved_mae[model], marker="o", label=model.replace("_", " ").title())
ax.set(title="Forecast test MAE by horizon", xlabel="Half-hour slots ahead", ylabel="MAE (occupancy rate)")
ax.set_xticks(saved_mae.index)
ax.legend()
ax.grid(alpha=.25)
fig.tight_layout()
st.pyplot(fig, clear_figure=True)

rank_rows = []
for horizon, row in saved_mae.iterrows():
    ordered = row.dropna().sort_values()
    winner = str(ordered.index[0])
    runner = str(ordered.index[1]) if len(ordered) > 1 else "—"
    relative = (1 - float(ordered.iloc[0]) / float(ordered.iloc[1])) * 100 if len(ordered) > 1 and ordered.iloc[1] else np.nan
    rank_rows.append({"Slots ahead": int(horizon), "Best model": winner,
                      "Best MAE": float(ordered.iloc[0]), "Runner-up": runner,
                      "Runner-up MAE": float(ordered.iloc[1]) if len(ordered) > 1 else np.nan,
                      "MAE improvement vs runner-up (%)": relative})
ranking = pd.DataFrame(rank_rows)

st.subheader("Ranking summary and model selection")
st.dataframe(ranking.style.format({"Best MAE": "{:.5f}", "Runner-up MAE": "{:.5f}",
                                   "MAE improvement vs runner-up (%)": "{:.2f}%"}),
             hide_index=True, use_container_width=True)
best_overall = ranking["Best model"].value_counts().idxmax()
best_at_one = ranking.loc[ranking["Slots ahead"] == 1].iloc[0]
st.success(f"**Best saved model:** {best_overall.replace('_', ' ').title()} has the lowest saved test MAE at "
           f"{int((ranking['Best model'] == best_overall).sum())} of {len(ranking)} horizons. "
           f"At one slot ahead it scores {best_at_one['Best MAE']:.5f} versus "
           f"{best_at_one['Runner-up']} at {best_at_one['Runner-up MAE']:.5f}.")
xgb_wins = int((ranking["Best model"] == "xgboost").sum())
if xgb_wins == len(ranking):
    st.write(f"**Why XGBoost is the selected production model:** it has the lowest saved test MAE at all {len(ranking)} horizons. "
             f"At one slot ahead it improves on the runner-up by {best_at_one['MAE improvement vs runner-up (%)']:.2f}%; "
             f"the six-slot forecast's runner-up margin is "
             f"{ranking.loc[ranking['Slots ahead'] == 6, 'MAE improvement vs runner-up (%)'].iloc[0]:.2f}%. "
             "The persisted bundle also supplies all six required forecast steps.")
else:
    st.warning(f"The current production pipeline uses XGBoost, but the saved MAE leaderboard ranks it first at "
               f"{xgb_wins} of {len(ranking)} horizons. The lowest-error model overall is {best_overall.replace('_', ' ').title()}; "
               "the current implementation and empirical ranking therefore differ at some horizons.")
st.caption("Selection here follows held-out MAE; it does not imply the winner is best under every metric or that forecast skill proves pricing impact.")

st.subheader("Full regression / forecasting metrics")
st.write("Full metrics below are computed by applying the already-fitted saved XGBoost models and naive persistence baseline to the actual held-out test rows. No model is refit. Random Forest and ARIMA provide only their saved MAE because their test predictions were not stored.")
if frozen_forecast_metrics is None:
    st.warning("The running app has an older cached analytics module. Restart Streamlit to load the current full-metrics helper; the saved-MAE leaderboard above is still available.")
try:
    frozen = load_frozen_metrics()
except Exception as exc:
    frozen = pd.DataFrame()
    st.warning(f"Could not compute additional metrics from the frozen model: {exc}")

if not frozen.empty:
    horizon = st.selectbox("Forecast horizon", sorted(frozen.horizon.unique()),
                           format_func=lambda value: f"{value} slot(s) ahead ({value*30} minutes)")
    detailed = frozen[frozen.horizon == horizon].copy()
    persisted = saved_mae.loc[horizon]
    rows = []
    for model in saved_mae.columns:
        match = detailed[detailed.model == ("XGBoost (saved)" if model == "xgboost" else "Naive")]
        if model in ("xgboost", "naive") and not match.empty:
            row = match.iloc[0].to_dict()
            row["Saved comparison MAE"] = float(persisted[model])
            row["Model"] = row.pop("model")
            rows.append(row)
        else:
            rows.append({"Model": model, "horizon": int(horizon), "n": np.nan,
                         "MAE": float(persisted[model]), "RMSE": np.nan, "R²": np.nan,
                         "MAPE (%)": np.nan, "SMAPE (%)": np.nan,
                         "MAPE denominator rows": np.nan,
                         "source": "Saved MAE only; predictions not persisted",
                         "Saved comparison MAE": float(persisted[model])})
    metrics_table = pd.DataFrame(rows)
    columns = ["Model", "n", "MAE", "RMSE", "R²", "MAPE (%)", "SMAPE (%)",
               "Saved comparison MAE", "source"]
    st.dataframe(metrics_table[columns].rename(columns={"n": "Test rows", "source": "Metric source"})
                 .style.format({"MAE": "{:.5f}", "RMSE": "{:.5f}", "R²": "{:.4f}",
                                "MAPE (%)": "{:.2f}%", "SMAPE (%)": "{:.2f}%",
                                "Saved comparison MAE": "{:.5f}"}, na_rep="Not available"),
                 hide_index=True, use_container_width=True)
    st.caption("MAPE excludes zero actual targets (the table reports the included denominator count); SMAPE uses a symmetric denominator and returns zero when both values are zero. R² is supplementary for forecast targets.")
    st.caption("The canonical leaderboard uses train.py's saved test scores from the validation-selected XGBoost model. The additional XGBoost metrics above use the persisted production model refit on train + validation, so its recomputed MAE can differ slightly from the canonical score.")

    st.markdown("#### Radar comparison · normalized one-horizon performance")
    radar_data = frozen[frozen.horizon == horizon].set_index("model")
    metric_names = ["MAE", "RMSE", "MAPE (%)", "SMAPE (%)", "R²"]
    available_models = [name for name in radar_data.index
                        if radar_data.loc[name, metric_names].notna().all()]
    if len(available_models) >= 2:
        values = radar_data.loc[available_models, metric_names]
        scores = pd.DataFrame(index=values.index, columns=metric_names, dtype=float)
        for metric in metric_names:
            low, high = float(values[metric].min()), float(values[metric].max())
            if np.isclose(low, high):
                scores[metric] = 1.0
            elif metric == "R²":
                scores[metric] = (values[metric] - low) / (high - low)
            else:
                scores[metric] = (high - values[metric]) / (high - low)
        angles = np.linspace(0, 2*np.pi, len(metric_names), endpoint=False).tolist()
        angles += angles[:1]
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, polar=True)
        for model, row in scores.iterrows():
            points = row.to_list() + [row.iloc[0]]
            ax.plot(angles, points, linewidth=2, label=model)
            ax.fill(angles, points, alpha=.08)
        ax.set_xticks(angles[:-1], metric_names)
        ax.set_ylim(0, 1)
        ax.set_title(f"Relative metric score · horizon {horizon}")
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True)
        st.caption("Radar axes are scaled within the models with complete metrics at the selected horizon. A score nearer 1 is better; it is a relative visualization, not an additional evaluation metric.")
    else:
        st.info("The radar requires at least two models with complete, comparable metrics. Current saved predictions support full metrics for XGBoost and the naive baseline only.")

st.subheader("Clustering model · separate evaluation")
saved = load_evaluation_metrics()
silhouette = saved.get("kmeans_silhouette_k4")
if silhouette is not None:
    st.metric("K-Means silhouette score (k=4)", f"{float(silhouette):.3f}")
    st.caption("K-Means is an unsupervised four-cluster demand-label model. Silhouette is not comparable to regression or forecasting error metrics.")
else:
    st.info("No saved K-Means evaluation was found.")
