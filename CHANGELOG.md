# Changelog

All notable changes to the DataKit project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-08-09

### Fixed
- **Smart Model & Task Auto-Detection in `data.fit()`**: Added explicit model task overrides for `"linear"`, `"ridge"`, `"lasso"`, and `"svr"` so regression models automatically map to regression task even on small sample datasets.

## [0.2.1] - 2026-08-09

### Fixed
- **Python 3.12 Type Annotation Compatibility**: Added `from __future__ import annotations` to `datakit.py` to prevent `NameError: name 'DataKit' is not defined` when evaluating class method return annotations in Python 3.12+ runtime environments.

## [0.2.0] - 2026-08-09

### Added
- **Approved PRD Ergonomic Additions**:
  - `data.correlations(target)`: Filtered feature correlations sorted by absolute magnitude against target.
  - `data.duplicates(subset=None)`: Standalone non-mutating view returning DataFrame of duplicate rows.
  - `data.compare(other)`: Structural diff (`CompareResult`) analyzing shape, column, and dtype changes between DataKits.
  - `dk.safe.reshape_column(series)`: Explicit 1D to 2D column vector shape converter.
  - `data.plot.kde(column)`: Standalone Kernel Density Estimate plot without histogram bars.

## [0.1.0] - 2026-08-08

### Added
- **Core Abstraction Layer (`DataKit`)**: Single container wrapping Pandas DataFrame with copy semantics and non-destructive operations.
- **Safety Operations (`dk.safe.*`)**:
  - `subtract()`, `add()`, `multiply()`, `divide()` with rank-mismatch guards flagging `ImplicitBroadcastWarning` or raising `ShapeMismatchError` in strict mode.
  - `check_shapes()` for pre-flight shape compatibility checking.
  - `is_view()` for array memory slice detection.
  - `align_check()` for index alignment verification flagging `IndexAlignmentWarning`.
  - `reshape_column()` for explicit 1D to 2D shape reshaping.
- **Data Quality & Audit (`data.audit()`, `data.clean()`, `data.duplicates()`)**:
  - `audit()`: Automated quality health scan identifying missingness, duplicates, high cardinality, near-constants, target leakage, and ID-like columns.
  - `clean()`: Non-destructive data cleaning with `confirm=True` confirmation gate, returning a new `DataKit` instance.
  - `duplicates()`: Standalone view of duplicate rows for inspection.
- **Exploratory Data Analysis (`data.inspect()`, `data.distributions()`, `data.outliers()`, `data.relationships()`, `data.correlations()`, `data.compare()`, `data.eda()`)**:
  - `inspect()`: Fast structural dataset overview.
  - `distributions()`: Skewness, kurtosis, and normality indicators for numeric columns.
  - `outliers()`: Outlier detection via IQR multiplier or Z-score thresholding.
  - `relationships()`: Pairwise Pearson/Spearman correlation matrix and strong pair extraction.
  - `correlations()`: Sorted feature correlations against a specified target column.
  - `compare()`: Structural diff between two DataFrames analyzing shape, column, and dtype changes.
  - `eda()`: Comprehensive synthesis pipeline aggregating inspect, audit, distribution, outlier, and relationship findings.
- **Object-Oriented Visualization (`data.plot.*`)**:
  - `hist()`, `box()`, `scatter()`, `bar()`, `count()`, `line()`, `heatmap()`, `violin()`, `pairplot()`, `kde()`.
  - 4-level precedence resolution (call kwarg > instance style > global config > default).
  - External `ax=` subplot injection support.
- **Machine Learning Preparation (`data.prepare()`)**:
  - Reproducible train/test splits (`X_train`, `X_test`, `y_train`, `y_test`) and fitted `sklearn.pipeline.Pipeline`.
  - Feature scaling (`StandardScaler`) and encoding (`OneHotEncoder` / `OrdinalEncoder`).
  - Target leakage detection and index alignment verification.
- **Synthesis Reporting (`data.report()`)**:
  - Export 8-section synthesis reports in HTML, Markdown, or JSON formats using hand-rolled zero-dependency templates.
- **Global Configuration (`dk.config`)**:
  - Runtime threshold configuration singleton (`set()`, `get()`, `reset()`).
