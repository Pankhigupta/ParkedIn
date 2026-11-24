"""Reusable exploratory-data-analysis summaries (no plotting side effects)."""
from __future__ import annotations

import pandas as pd


def missing_value_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Return non-empty columns' missing counts and proportions, sorted by count."""
    counts = data.isna().sum()
    result = pd.DataFrame({"missing_count": counts, "missing_percent": counts / max(len(data), 1) * 100})
    return result.sort_values(["missing_count", "missing_percent"], ascending=False)


def numeric_summary(data: pd.DataFrame) -> pd.DataFrame:
    """Summarize numeric columns with count, spread, quartiles, and extrema."""
    numeric = data.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()
    return numeric.describe(percentiles=[.25, .5, .75]).transpose()


def categorical_summary(data: pd.DataFrame, max_categories: int = 20) -> dict[str, pd.Series]:
    """Return value counts for non-numeric columns, bounded for review-friendly output."""
    categorical = data.select_dtypes(exclude="number")
    return {column: categorical[column].value_counts(dropna=False).head(max_categories)
            for column in categorical.columns}


def dataset_overview(data: pd.DataFrame) -> dict:
    """Summarize data shape, types, duplicate rows, and missingness."""
    return {
        "shape": data.shape,
        "dtypes": data.dtypes.astype(str).to_dict(),
        "duplicate_rows": int(data.duplicated().sum()),
        "missing": missing_value_summary(data),
        "numeric": numeric_summary(data),
    }
