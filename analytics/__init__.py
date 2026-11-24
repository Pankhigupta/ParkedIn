"""Read-only academic analytics helpers for ParkedIn."""

from .statistics import (dataset_catalog, load_dataset, load_simulation_table,
                         model_catalog)

__all__ = ["dataset_catalog", "load_dataset", "load_simulation_table", "model_catalog"]
