"""Built-in synthetic datasets module for DataKit.
"""
from __future__ import annotations

import difflib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datakit.core.datakit import DataKit

_DATA_DIR = Path(__file__).parent / "data"

_DATASETS = {
    "insurance": "insurance.csv",
    "housing": "housing.csv",
    "churn": "churn.csv",
}


def list_datasets() -> list[str]:
    """List all available built-in synthetic datasets.

    Returns:
        Sorted list of dataset names.
    """
    return sorted(list(_DATASETS.keys()))


def load_dataset(name: str) -> DataKit:
    """Load a built-in synthetic dataset safely offline into a DataKit instance.

    Args:
        name: Name of dataset (e.g. 'insurance', 'housing', 'churn').

    Returns:
        DataKit object wrapping the synthetic dataset.

    Raises:
        ValueError: If dataset name is unknown.
    """
    clean_name = str(name).strip().lower()
    if clean_name not in _DATASETS:
        matches = difflib.get_close_matches(clean_name, _DATASETS.keys())
        suggestion = f" Did you mean '{matches[0]}'?" if matches else ""
        avail = ", ".join(f"'{k}'" for k in sorted(_DATASETS.keys()))
        raise ValueError(
            f"Dataset '{name}' not found.{suggestion} Available datasets: [{avail}]"
        )

    file_path = _DATA_DIR / _DATASETS[clean_name]
    from datakit.core.datakit import DataKit

    return DataKit(file_path)
