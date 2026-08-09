import html
import warnings
from pathlib import Path
from typing import Any, Literal
import numpy as np
import pandas as pd

from datakit.core.exceptions import EmptyDataError
from datakit.core.warnings import LargeDatasetWarning
from datakit.core.results import (
    AuditResult,
    CleanReport,
    CompareResult,
    DistributionResult,
    EDAResult,
    InspectResult,
    ModelResult,
    OutlierResult,
    PlotResult,
    PrepareResult,
    RelationshipResult,
)
from datakit.config import config


class DataKit:
    """The main DataKit class for exploratory data analysis and data preparation."""

    def __init__(self, source: str | Path | pd.DataFrame | dict[str, Any]):
        if isinstance(source, pd.DataFrame):
            self._df = source.copy()
        elif isinstance(source, dict):
            self._df = pd.DataFrame(source)
        elif isinstance(source, (str, Path)):
            path_str = str(source).lower()
            if path_str.endswith('.csv'):
                self._df = pd.read_csv(source)
            elif path_str.endswith(('.xlsx', '.xls')):
                try:
                    self._df = pd.read_excel(source)
                except ImportError as e:
                    raise ImportError("openpyxl is required to read Excel files.") from e
            elif path_str.endswith('.parquet'):
                try:
                    self._df = pd.read_parquet(source)
                except ImportError as e:
                    raise ImportError("pyarrow or fastparquet is required to read Parquet files.") from e
            elif path_str.endswith('.json'):
                self._df = pd.read_json(source)
            else:
                raise ValueError(f"Unsupported file format for source: {source}")
        else:
            raise TypeError(
                f"DataKit expects a file path (str/Path), DataFrame, or dict. "
                f"Got {type(source).__name__}."
            )
        self._plot: Any = None
        self._last_clean_report: CleanReport | None = None

    @property
    def last_clean_report(self) -> CleanReport | None:
        """Return the CleanReport from the most recent clean() call on this instance, if any."""
        return self._last_clean_report

    @property
    def df(self) -> pd.DataFrame:
        """Return a live reference to the internal DataFrame."""
        return self._df

    def inspect(self, n: int = 5, memory: bool = True) -> InspectResult:
        """Provide a fast structural overview of the dataset.

        Args:
            n: Number of rows to include in head/tail preview (default: 5).
            memory: Whether to calculate deep memory usage (default: True).

        Returns:
            InspectResult object.

        Raises:
            EmptyDataError: If the DataFrame has 0 rows.
        """
        if self._df.empty:
            raise EmptyDataError()

        shape = (int(self._df.shape[0]), int(self._df.shape[1]))
        dtypes = self._df.dtypes

        if memory:
            mem_bytes = self._df.memory_usage(deep=True).sum()
            memory_mb = float(mem_bytes / (1024 * 1024))
            large_threshold = config.get("large_dataset_mb")
            if memory_mb > large_threshold:
                warnings.warn(
                    f"Dataset memory size ({memory_mb:.1f} MB) exceeds threshold ({large_threshold} MB). "
                    f"Calculating deep memory usage can be slow on large datasets; pass memory=False to skip.",
                    LargeDatasetWarning,
                )
        else:
            memory_mb = 0.0

        head = self._df.head(n)
        tail = self._df.tail(n)

        index_info = {
            "type": type(self._df.index).__name__,
            "is_monotonic": bool(self._df.index.is_monotonic_increasing),
            "has_duplicates": bool(self._df.index.has_duplicates),
        }

        return InspectResult(
            shape=shape,
            dtypes=dtypes,
            memory_mb=memory_mb,
            head=head,
            tail=tail,
            index_info=index_info,
        )

    def audit(
        self,
        checks: list[str] | None = None,
        severity_threshold: Literal["info", "warning", "critical"] = "info",
    ) -> AuditResult:
        """Perform a structured, prioritized data-quality audit with recommendations.

        Args:
            checks: List of check names to run, or None for all.
                Supported: "missing", "duplicates", "constant", "cardinality",
                "suspicious_dtype", "id_like", "leakage".
            severity_threshold: Minimum severity ("info", "warning", "critical") to return.

        Returns:
            AuditResult object.
        """
        from datakit.quality.audit import audit as _audit

        return _audit(self._df, checks=checks, severity_threshold=severity_threshold)

    def clean(
        self,
        missing: Literal["report", "drop", "impute_mean", "impute_median", "impute_mode"] = "report",
        duplicates: Literal["report", "drop"] = "report",
        dtypes: Literal["report", "coerce"] = "report",
        confirm: bool = False,
    ) -> DataKit:
        """Guided, reported, non-silent cleaning of missing values, duplicates, and dtypes.

        Returns a NEW DataKit instance wrapping a new DataFrame. Never mutates original data.

        Args:
            missing: Missing-value strategy ("report", "drop", "impute_mean", "impute_median", "impute_mode").
            duplicates: Duplicate-row strategy ("report", "drop").
            dtypes: Dtype coercion strategy ("report", "coerce").
            confirm: Must be True to execute any non-report strategy.

        Returns:
            New DataKit instance with .last_clean_report attached.

        Raises:
            ConfirmationRequiredError: If destructive strategy requested without confirm=True.
        """
        from datakit.quality.clean import clean_dataframe

        new_df, report = clean_dataframe(
            self._df,
            missing=missing,
            duplicates=duplicates,
            dtypes=dtypes,
            confirm=confirm,
        )

        new_datakit = DataKit(new_df)
        new_datakit._last_clean_report = report
        return new_datakit

    def outliers(
        self,
        columns: list[str] | None = None,
        method: Literal["iqr", "zscore"] = "iqr",
        threshold: float | None = None,
    ) -> OutlierResult:
        """Detect and report outliers per numeric column using IQR or Z-score methods.

        Args:
            columns: Subset of numeric column names to analyze, or None for all.
            method: Outlier method ("iqr" or "zscore").
            threshold: IQR multiplier (default: 1.5) or z-score cutoff (default: 3.0).

        Returns:
            OutlierResult object.
        """
        from datakit.analysis.outliers import detect_outliers

        return detect_outliers(self._df, columns=columns, method=method, threshold=threshold)

    def relationships(
        self,
        method: Literal["pearson", "spearman"] = "pearson",
        threshold: float = 0.7,
    ) -> RelationshipResult:
        """Analyze correlations and flag strong pairwise relationships.

        Args:
            method: Correlation method ("pearson" or "spearman").
            threshold: Absolute correlation threshold to flag as strong pair (0.0 to 1.0).

        Returns:
            RelationshipResult object.
        """
        from datakit.analysis.relationships import analyze_relationships

        return analyze_relationships(self._df, method=method, threshold=threshold)

    def correlations(
        self,
        target: str,
        method: Literal["pearson", "spearman"] = "pearson",
    ) -> pd.Series:
        """Calculate sorted feature correlations against a specific target column.

        Args:
            target: Name of target column.
            method: Correlation method ("pearson" or "spearman").

        Returns:
            pd.Series sorted by absolute correlation magnitude.
        """
        from datakit.analysis.relationships import get_target_correlations

        return get_target_correlations(self._df, target=target, method=method)

    def duplicates(
        self,
        subset: list[str] | str | None = None,
    ) -> pd.DataFrame:
        """Return a new DataFrame containing all duplicate rows for inspection.

        Args:
            subset: Column name or list of column names to consider for duplicate checking.

        Returns:
            New pd.DataFrame containing duplicate rows.
        """
        from datakit.quality.audit import get_duplicate_rows

        return get_duplicate_rows(self._df, subset=subset)

    def compare(
        self,
        other: DataKit | pd.DataFrame,
    ) -> CompareResult:
        """Compare this dataset with another DataKit or DataFrame to highlight structural changes.

        Args:
            other: Second DataKit instance or DataFrame to compare against.

        Returns:
            CompareResult object.
        """
        from datakit.core.infer import compare_datasets

        other_df = other.df if isinstance(other, DataKit) else other
        return compare_datasets(self._df, other_df)

    def distributions(
        self,
        columns: list[str] | None = None,
    ) -> DistributionResult:
        """Summarize distribution shapes (skewness, kurtosis, normality flag) per numeric column.

        Args:
            columns: Subset of numeric column names to analyze, or None for all.

        Returns:
            DistributionResult object.
        """
        from datakit.analysis.distributions import analyze_distributions

        return analyze_distributions(self._df, columns=columns)

    def eda(
        self,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
        plots: bool = True,
        target: str | None = None,
        sample: int | None = None,
        strict: bool = False,
    ) -> EDAResult:
        """Execute a full exploratory data analysis pipeline synthesis.

        Args:
            include: List of steps to include ("inspect", "audit", "distributions", "outliers", "relationships", "plots").
            exclude: List of steps to exclude.
            plots: Whether to generate default plot visualizations.
            target: Optional target column name to bias analysis.
            sample: Optional random sample size (int) to use.
            strict: If True, step errors raise immediately; if False, fails soft.

        Returns:
            EDAResult composite object.
        """
        from datakit.eda.eda import run_eda

        return run_eda(
            self._df,
            include=include,
            exclude=exclude,
            plots=plots,
            target=target,
            sample=sample,
            strict=strict,
        )

    def visualize(self) -> list[PlotResult]:
        """Auto-plot reasonable default visualizations for the dataset.

        Returns:
            List of PlotResult objects.
        """
        from datakit.eda.eda import visualize as _visualize

        return _visualize(self._df)

    def prepare(
        self,
        target: str,
        task: Literal["classification", "regression"] = "classification",
        test_size: float = 0.2,
        random_state: int | None = None,
        scale: bool = True,
        encode: Literal["onehot", "ordinal", "none"] = "onehot",
        strict_leakage: bool = False,
    ) -> PrepareResult:
        """Prepare dataset for machine learning models with scikit-learn pipelines.

        Args:
            target: Name of target column (required).
            task: Task type ("classification" or "regression").
            test_size: Proportion of dataset for test split (default: 0.2).
            random_state: Seed for reproducible train/test split.
            scale: Whether to apply StandardScaler to numeric features.
            encode: Categorical encoding ("onehot", "ordinal", "none").
            strict_leakage: If True, raises DataKitError on target leakage correlation; if False, warns.

        Returns:
            PrepareResult object containing train/test splits and fitted sklearn Pipeline.

        Raises:
            ImportError: If scikit-learn is not installed.
            ColumnNotFoundError: If target column is missing.
        """
        from datakit.ml.prepare import prepare_data

        return prepare_data(
            self._df,
            target=target,
            task=task,
            test_size=test_size,
            random_state=random_state,
            scale=scale,
            encode=encode,
            strict_leakage=strict_leakage,
        )

    def encode_target(self, target: str) -> tuple[pd.Series, dict[Any, int]]:
        """Encode a categorical target column into 0-indexed integer labels.

        Args:
            target: Name of target column to encode.

        Returns:
            Tuple of (encoded Series, reverse label mapping dictionary).
        """
        from datakit.ml.ml_helpers import encode_target_labels

        return encode_target_labels(self._df, target=target)

    def imbalance_ratio(self, target: str) -> pd.DataFrame:
        """Analyze target class distribution and class imbalance ratios.

        Args:
            target: Classification target column name.

        Returns:
            pd.DataFrame with class counts, percentages, and imbalance ratios.
        """
        from datakit.ml.ml_helpers import check_imbalance

        return check_imbalance(self._df, target=target)

    def cv_splits(
        self,
        target: str | None = None,
        n_splits: int = 5,
        stratified: bool = True,
        random_state: int | None = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate K-Fold or Stratified K-Fold cross-validation index splits.

        Args:
            target: Optional target column for stratified splitting.
            n_splits: Number of CV folds (default: 5).
            stratified: Whether to use Stratified K-Fold when target is provided.
            random_state: Seed for reproducible splits.

        Returns:
            List of (train_indices, validation_indices) tuples.
        """
        from datakit.ml.ml_helpers import create_cv_splits

        return create_cv_splits(
            self._df,
            target=target,
            n_splits=n_splits,
            stratified=stratified,
            random_state=random_state,
        )

    def fit(
        self,
        target: str,
        model: str = "rf",
        task: Literal["classification", "regression", "auto"] = "auto",
        test_size: float = 0.2,
        scale: bool = True,
        encode: Literal["onehot", "ordinal", "none"] = "onehot",
        random_state: int | None = 42,
        **model_kwargs: Any,
    ) -> ModelResult:
        """Train and evaluate any machine learning algorithm in one simple step.

        Args:
            target: Name of target column.
            model: Algorithm shortcut ("rf", "linear", "logistic", "tree", "svc", "svr", "gb", "extra_trees", "knn", "ridge", "lasso", "naive_bayes").
            task: Task type ("classification", "regression", or "auto").
            test_size: Test set proportion (default: 0.2).
            scale: Whether to scale numeric features.
            encode: Categorical encoding ("onehot", "ordinal", "none").
            random_state: Random seed for training.
            **model_kwargs: Additional parameters passed to scikit-learn estimator.

        Returns:
            ModelResult object containing trained model, metrics, and evaluation plots.
        """
        from datakit.ml.models import fit_model

        return fit_model(
            self._df,
            target=target,
            model=model,
            task=task,
            test_size=test_size,
            scale=scale,
            encode=encode,
            random_state=random_state,
            **model_kwargs,
        )

    def report(
        self,
        result: EDAResult | AuditResult | None = None,
        format: Literal["html", "markdown", "json"] = "html",
        path: str | Path | None = None,
    ) -> str:
        """Export a structured 8-section report in HTML, Markdown, or JSON.

        Args:
            result: Pre-computed EDAResult or AuditResult, or None to run eda() fresh.
            format: Export format ("html", "markdown", "json").
            path: File path to save report.

        Returns:
            Path string of saved report file.

        Raises:
            ValueError: If format is 'pdf' or unsupported.
        """
        from datakit.reporting.report import generate_report

        return generate_report(self, result=result, format=format, path=path)

    @property
    def plot(self) -> Any:
        if self._plot is None:
            try:
                from datakit.visualization.plot import PlotNamespace
                self._plot = PlotNamespace(self)
            except ImportError as e:
                raise ImportError(
                    "Could not import datakit.visualization.plot. "
                    "Make sure the visualization module is correctly set up."
                ) from e
        return self._plot

    def __repr__(self) -> str:
        rows, cols = self._df.shape
        mem = self._df.memory_usage(deep=True).sum() / (1024 * 1024)
        return f"DataKit(rows={rows}, columns={cols}, memory={mem:.2f}MB)"

    def _repr_html_(self) -> str:
        rows, cols = self._df.shape
        mem = self._df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        html = [
            "<div style='font-family: sans-serif;'>",
            f"<h4>DataKit Dataset</h4>",
            f"<p><strong>Shape:</strong> {rows} rows &times; {cols} columns</p>",
            f"<p><strong>Memory:</strong> {mem:.2f} MB</p>",
            "<h5>Data Types Summary:</h5>",
            self._df.dtypes.value_counts().to_frame(name="Count")._repr_html_(),
            "<h5>Preview (first 3 rows):</h5>",
            self._df.head(3)._repr_html_(),
            "</div>"
        ]
        return "\n".join(html)


def read(source: str | Path | pd.DataFrame | dict[str, Any]) -> DataKit:
    """Auto-detect file format from extension (.csv, .xlsx, .parquet, .json) and load into DataKit.

    Args:
        source: File path (str/Path), DataFrame, or dict.

    Returns:
        DataKit instance.
    """
    return DataKit(source)
