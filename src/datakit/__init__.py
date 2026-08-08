"""
DataKit is a Python library for fast exploratory data analysis and data preparation.

Example:
    >>> from datakit import DataKit
    >>> dk = DataKit("data.csv")
    >>> dk.df.head()
"""

from datakit.core.datakit import DataKit, read
from datakit.config import config
from datakit.datasets import list_datasets, load_dataset
import datakit.safety.safe_ops as safe

__version__ = "0.1.0"
__all__ = ["DataKit", "read", "config", "safe", "load_dataset", "list_datasets"]
