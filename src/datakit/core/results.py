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


@dataclass
class ModelResult(DataKitResult):
    model_name: str
    task: str
    model: Any
    metrics: dict[str, float]
    feature_importances: pd.Series | None
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    y_pred: pd.Series

    def summary(self) -> str:
        lines = [
            f"=== DataKit Model Training Report ({self.model_name}) ===",
            f"Task Type: {self.task.upper()}",
            "Evaluation Metrics:",
        ]
        for name, val in self.metrics.items():
            lines.append(f"  - {name.upper()}: {val:.4f}")

        if self.feature_importances is not None and not self.feature_importances.empty:
            lines.append("\nTop 5 Feature Importances:")
            top5 = self.feature_importances.head(5)
            for feat, val in top5.items():
                lines.append(f"  - {feat}: {val:.4f}")

        return "\n".join(lines)

    def predict(self, X: pd.DataFrame | np.ndarray) -> pd.Series:
        preds = self.model.predict(X)
        return pd.Series(preds)

    def plot_importance(self, top_n: int = 10) -> PlotResult:
        if self.feature_importances is None or self.feature_importances.empty:
            raise ValueError(
                f"Model '{self.model_name}' does not support feature importances. "
                "Feature importance is only available for tree-based models (e.g., RandomForest, DecisionTree, ExtraTrees, GradientBoosting) "
                "or models with computed feature importances."
            )
        import matplotlib.pyplot as plt
        import seaborn as sns

        top_feats = self.feature_importances.head(top_n).sort_values(ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=top_feats.values, y=top_feats.index, hue=top_feats.index, palette="viridis", legend=False, ax=ax)
        ax.set_title(f"Top {len(top_feats)} Feature Importances ({self.model_name})")
        ax.set_xlabel("Importance Score")
        return PlotResult(fig=fig, ax=ax, call_info="plot_importance()")

    def plot_evaluation(self) -> PlotResult:
        import matplotlib.pyplot as plt

        if self.task == "classification":
            import seaborn as sns
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(self.y_test, self.y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            ax.set_title(f"Confusion Matrix ({self.model_name})")
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Label")
            return PlotResult(fig=fig, ax=ax, call_info="plot_evaluation()")
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(self.y_test, self.y_pred, alpha=0.7, color="#0066cc")
            min_val = min(float(self.y_test.min()), float(self.y_pred.min()))
            max_val = max(float(self.y_test.max()), float(self.y_pred.max()))
            ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Ideal Perfect Fit")
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title(f"Actual vs. Predicted ({self.model_name})")
            ax.legend()
            return PlotResult(fig=fig, ax=ax, call_info="plot_evaluation()")
