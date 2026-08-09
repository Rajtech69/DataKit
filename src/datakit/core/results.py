from __future__ import annotations
import html
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, TYPE_CHECKING
import pandas as pd
import numpy as np

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


@dataclass
class DataKitResult:
    """Base class for all DataKit results."""

    def to_dict(self) -> dict[str, Any]:
        """Convert all fields to a dictionary."""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, DataKitResult):
                result[key] = value.to_dict()
            elif isinstance(value, pd.DataFrame):
                result[key] = value.to_dict(orient="records")
            elif isinstance(value, pd.Series):
                result[key] = value.to_dict()
            elif isinstance(value, np.ndarray):
                result[key] = value.tolist()
            else:
                result[key] = value
        return result

    def _repr_html_(self) -> str:
        """Default HTML representation for Jupyter environments."""
        html_str = ["<table style='border: 1px solid black; border-collapse: collapse;'>"]
        for key, value in self.__dict__.items():
            val_str = html.escape(str(value))
            html_str.append(
                f"<tr><th style='border: 1px solid black; padding: 5px; text-align: left;'>{html.escape(key)}</th>"
                f"<td style='border: 1px solid black; padding: 5px;'>{val_str}</td></tr>"
            )
        html_str.append("</table>")
        return "".join(html_str)

    def __repr__(self) -> str:
        """Simple text representation."""
        fields = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"{self.__class__.__name__}({fields})"


@dataclass
class Issue:
    column: str | None
    severity: str
    message: str
    recommendation: str


@dataclass
class InspectResult(DataKitResult):
    shape: tuple[int, int]
    dtypes: pd.Series
    memory_mb: float
    head: pd.DataFrame
    tail: pd.DataFrame
    index_info: dict[str, Any]

    def _repr_html_(self) -> str:
        html_str = [
            f"<div><strong>InspectResult</strong></div>",
            f"<ul>",
            f"<li><strong>Shape:</strong> {self.shape[0]} rows, {self.shape[1]} columns</li>",
            f"<li><strong>Memory Usage:</strong> {self.memory_mb:.2f} MB</li>",
            f"</ul>",
            f"<div><strong>Data Types:</strong></div>",
            self.dtypes.to_frame(name="dtype")._repr_html_(),
            f"<div><strong>Head:</strong></div>",
            self.head._repr_html_(),
            f"<div><strong>Tail:</strong></div>",
            self.tail._repr_html_()
        ]
        return "".join(html_str)


@dataclass
class AuditResult(DataKitResult):
    summary: str
    issues: list[Issue]

    @property
    def warnings(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "warning"]

    @property
    def critical(self) -> list[Issue]:
        return [issue for issue in self.issues if issue.severity == "critical"]

    @property
    def recommendations(self) -> list[str]:
        recs = []
        for issue in self.issues:
            if issue.recommendation not in recs:
                recs.append(issue.recommendation)
        return recs

    def to_dataframe(self) -> pd.DataFrame:
        data = [
            {
                "column": issue.column,
                "severity": issue.severity,
                "message": issue.message,
                "recommendation": issue.recommendation
            }
            for issue in self.issues
        ]
        return pd.DataFrame(data)

    def _repr_html_(self) -> str:
        colors = {"critical": "red", "warning": "orange", "info": "blue"}
        html_str = [
            "<div><h4>AuditResult</h4>",
            f"<strong>Summary:</strong> {html.escape(self.summary)}</div>",
            "<table style='border: 1px solid black; border-collapse: collapse;'>",
            "<tr><th style='border: 1px solid black; padding: 5px;'>Column</th>"
            "<th style='border: 1px solid black; padding: 5px;'>Severity</th>"
            "<th style='border: 1px solid black; padding: 5px;'>Message</th>"
            "<th style='border: 1px solid black; padding: 5px;'>Recommendation</th></tr>"
        ]
        for issue in self.issues:
            color = colors.get(issue.severity, "black")
            html_str.append(
                f"<tr style='color: {color};'>"
                f"<td style='border: 1px solid black; padding: 5px;'>{html.escape(str(issue.column))}</td>"
                f"<td style='border: 1px solid black; padding: 5px;'>{html.escape(issue.severity)}</td>"
                f"<td style='border: 1px solid black; padding: 5px;'>{html.escape(issue.message)}</td>"
                f"<td style='border: 1px solid black; padding: 5px;'>{html.escape(issue.recommendation)}</td>"
                "</tr>"
            )
        html_str.append("</table>")
        return "".join(html_str)


@dataclass
class CleanReport(DataKitResult):
    rows_dropped: int
    values_imputed: dict[str, int]
    dtype_changes: list[tuple[str, str, str]]
    original_shape: tuple[int, int]
    new_shape: tuple[int, int]

    def diff_summary(self) -> str:
        lines = [
            f"Shape changed from {self.original_shape} to {self.new_shape}",
            f"Rows dropped: {self.rows_dropped}",
        ]
        if self.values_imputed:
            lines.append("Values imputed:")
            for col, count in self.values_imputed.items():
                lines.append(f"  - {col}: {count}")
        if self.dtype_changes:
            lines.append("Data type changes:")
            for col, old, new in self.dtype_changes:
                lines.append(f"  - {col}: {old} -> {new}")
        return "\n".join(lines)


@dataclass
class OutlierResult(DataKitResult):
    counts: dict[str, int]
    indices: dict[str, list[int]]
    method: str
    threshold: float

    def summary(self) -> str:
        lines = [f"Outlier Detection Summary (Method: {self.method}, Threshold: {self.threshold})"]
        for col, count in self.counts.items():
            lines.append(f"  - {col}: {count} outliers")
        return "\n".join(lines)

    def to_dataframe(self, df: pd.DataFrame, flag_column: str = "_is_outlier") -> pd.DataFrame:
        new_df = df.copy()
        all_outlier_indices = set()
        for inds in self.indices.values():
            all_outlier_indices.update(inds)
        
        new_df[flag_column] = False
        new_df.loc[list(all_outlier_indices), flag_column] = True
        return new_df


@dataclass
class RelationshipResult(DataKitResult):
    matrix: pd.DataFrame
    strong_pairs: list[tuple[str, str, float]]
    method: str

    def summary(self) -> str:
        lines = [f"Relationship Summary (Method: {self.method})"]
        lines.append("Strong pairs:")
        for c1, c2, val in self.strong_pairs:
            lines.append(f"  - {c1} & {c2}: {val:.4f}")
        return "\n".join(lines)


@dataclass
class DistributionResult(DataKitResult):
    stats: pd.DataFrame

    def summary(self) -> str:
        return f"Distribution Summary:\n{self.stats.to_string()}"


@dataclass
class EDAResult(DataKitResult):
    inspect: InspectResult | None = None
    audit: AuditResult | None = None
    distributions: DistributionResult | None = None
    outliers: OutlierResult | None = None
    relationships: RelationshipResult | None = None
    figures: list[Any] = field(default_factory=list)
    warnings_collected: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = ["EDA Synthesis Report"]
        if self.inspect:
            parts.append(f"Shape: {self.inspect.shape}")
        if self.audit:
            parts.append(f"Critical Issues: {len(self.audit.critical)}")
            parts.append(f"Warnings: {len(self.audit.warnings)}")
        if self.outliers:
            total_outliers = sum(self.outliers.counts.values())
            parts.append(f"Total Outliers Detected: {total_outliers}")
        if self.relationships:
            parts.append(f"Strong Relationships: {len(self.relationships.strong_pairs)}")
        return "\n".join(parts)


@dataclass
class PrepareResult(DataKitResult):
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    preprocessing_pipeline: Any
    report: str


@dataclass
class CompareResult(DataKitResult):
    shape_a: tuple[int, int]
    shape_b: tuple[int, int]
    added_columns: list[str]
    removed_columns: list[str]
    common_columns: list[str]
    dtype_changes: list[tuple[str, str, str]]

    def summary(self) -> str:
        lines = [
            "=== Dataset Comparison Report ===",
            f"Dataset A Shape: {self.shape_a[0]} rows x {self.shape_a[1]} cols",
            f"Dataset B Shape: {self.shape_b[0]} rows x {self.shape_b[1]} cols",
            f"Added Columns ({len(self.added_columns)}): {self.added_columns}",
            f"Removed Columns ({len(self.removed_columns)}): {self.removed_columns}",
            f"Common Columns ({len(self.common_columns)}): {self.common_columns}",
        ]
        if self.dtype_changes:
            lines.append("Data Type Changes:")
            for col, old_t, new_t in self.dtype_changes:
                lines.append(f"  - {col}: {old_t} -> {new_t}")
        else:
            lines.append("Data Type Changes: None")
        return "\n".join(lines)


@dataclass
class PlotResult(DataKitResult):
    fig: Any
    ax: Any
    call_info: str

    @property
    def axes(self) -> list[Any]:
        if isinstance(self.ax, np.ndarray):
            return self.ax.flatten().tolist()
        elif isinstance(self.ax, list):
            return self.ax
        return [self.ax]

    def _repr_html_(self) -> str:
        return ""


@dataclass
class ShapeCheckResult(DataKitResult):
    shape_a: tuple[int, ...]
    shape_b: tuple[int, ...]
    compatible: bool
    is_implicit_broadcast: bool
    result_shape: tuple[int, ...] | None
    explanation: str


@dataclass
class AlignCheckResult(DataKitResult):
    overlapping_labels: list[Any]
    non_overlapping_df1: list[Any]
    non_overlapping_df2: list[Any]
    match_pct: float
    explanation: str
