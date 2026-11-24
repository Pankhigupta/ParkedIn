"""Research-oriented textual findings assembled from existing evaluation artifacts."""
from __future__ import annotations

from .model_evaluation import clustering_evaluation, feature_importance, forecast_comparison


def evaluation_findings() -> list[str]:
    """Create concise evidence-led observations; empty artifacts produce no claims."""
    findings = []
    forecasts = forecast_comparison()
    if not forecasts.empty:
        for horizon, row in forecasts.iterrows():
            valid = row.dropna()
            if not valid.empty:
                best = valid.idxmin()
                findings.append(
                    f"At {int(horizon)} slot(s) ahead, {best} has the lowest saved test MAE ({valid[best]:.4f}).")
    importance = feature_importance()
    if not importance.empty:
        findings.append(f"The largest saved one-step feature importance is {importance.index[0]} "
                        f"({importance.iloc[0]:.4f}).")
    cluster = clustering_evaluation()
    if cluster.get("silhouette_k4") is not None:
        findings.append(f"The saved four-cluster K-Means silhouette score is "
                        f"{float(cluster['silhouette_k4']):.3f}.")
    return findings
