"""DataKit is a safety-first, human-oriented abstraction layer over NumPy, Pandas, Matplotlib, and Seaborn.

Example:
    >>> import datakit as dk
    >>> data = dk.DataKit("data.csv")
    >>> data.inspect()
"""

from datakit.core.datakit import DataKit
from datakit.config import config
import datakit.safety.safe_ops as safe

__version__ = "0.3.0"
__all__ = ["DataKit", "config", "safe"]
