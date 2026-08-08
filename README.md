# DataKit: Safety-First Python Data Science Layer

[![PyPI Version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/datakit/)
[![Python Versions](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue)](https://pypi.org/project/datakit/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/tests-123%20passed-brightgreen.svg)](tests/)
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**DataKit** is a safety-first, human-oriented abstraction layer built on top of **NumPy**, **Pandas**, **Matplotlib**, and **Seaborn**. It does not replace any of them — instead, it exposes high-intent functions (`dk.DataKit()`, `data.audit()`, `data.clean()`, `dk.safe.*`, `data.plot.*`, `data.prepare()`, `data.report()`) that eliminate silent failures, state-bleed, and boilerplate code in data science workflows.

---

## 📖 Table of Contents

- [Main Features](#-main-features)
- [Installation](#-installation)
- [Quickstart](#-quickstart)
- [Architecture & Design Principles](#-architecture--design-principles)
- [User Guide & LaTeX Documentation](#-user-guide--latex-documentation)
- [Development & Testing](#-development--testing)
- [License](#-license)

---

## 🔥 Main Features

### 🛡️ Explicit Safety & Non-Silent Operations
- **Broadcasting Guard (`dk.safe.*`)**: Detects rank-mismatch outer-product generation (e.g., `(n,) - (n, 1) -> (n, n)`) before execution.
- **Guided Cleaning (`data.clean()`)**: Destructive operations (imputation, row dropping) require explicit `confirm=True`. Returns a new `DataKit` instance without mutating original DataFrames.
- **Data Loss Warnings**: Automatically issues `DataLossWarning` when operations drop >10% of rows.

### 📊 Pure Object-Oriented Visualizations (`data.plot.*`)
- Zero global `pyplot` state-bleed. All plots return a `PlotResult` with `.fig` and `.ax` properties.
- Supports 4-level precedence resolution (call-level > object-level > process config > default).
- Injection-level escape hatch accepts custom `ax=` for drawing into pre-existing `plt.subplots()` grids.

### 🤖 Machine Learning Dataset Preparation (`data.prepare()`)
- Automatic target leakage detection (|corr| >= 0.98).
- High-cardinality encoding warnings.
- Index-alignment verification between train/test splits.
- Wraps preprocessors into fitted `sklearn.pipeline.Pipeline` objects.

### 📄 Zero-Dependency Multi-Format Reporting (`data.report()`)
- Generates 8-section synthesis reports in standalone **HTML**, **Markdown**, and **JSON** without external template engines (PDF export explicitly disabled per PRD §7/§32).

---

## 📦 Installation

### From GitHub (Latest Development Version)
```bash
pip install git+https://github.com/Rajtech69/DataKit.git
```

With full optional dependencies (scikit-learn, openpyxl, pyarrow):
```bash
pip install "git+https://github.com/Rajtech69/DataKit.git#egg=datakit[all]"
```

### Pre-built Wheel Package
```bash
pip install dist/datakit-0.1.0-py3-none-any.whl
```

---

## ⚡ Quickstart

```python
import datakit as dk
import numpy as np

# 1. Auto-Detect & Load Data (dk.read supports CSV, Excel, Parquet, JSON)
data = dk.read("insurance.csv")

# 2. Inspect Structure & Audit Quality
print(data.inspect())
audit = data.audit()
print(audit.summary)

# 3. Non-Destructive Cleaning
cleaned = data.clean(missing="impute_median", duplicates="drop", confirm=True)
print(cleaned.last_clean_report.diff_summary())

# 4. Outliers & Correlation Analysis
outliers = cleaned.outliers(method="iqr")
print(outliers.summary())

# 5. Safe Arithmetic (Flags rank mismatch before executing)
a = np.arange(5)
b = np.arange(5)[:, None]
diff = dk.safe.subtract(a, b)

# 6. Object-Oriented Plotting
cleaned.plot.scatter("age", "charges", hue="smoker", trend=True)

# 7. Machine Learning Dataset Preparation
ml_splits = cleaned.prepare(target="charges", task="regression", scale=True, encode="onehot")
print("X_train shape:", ml_splits.X_train.shape)
print("Fitted Pipeline:", ml_splits.preprocessing_pipeline)

# 8. Export Standalone Report
cleaned.report(format="html", path="synthesis_report.html")
```

---

## 📐 Architecture & Design Principles

```
                              DataKit (Core Engine)
                                       │
    ┌──────────────────┬───────────────┼───────────────┬──────────────────┐
    ▼                  ▼               ▼               ▼                  ▼
 Data Quality     Broadcasting   Statistical EDA  Object-Oriented    ML Pipeline
  & Cleaning        Safety         & Profiling     Visualization     Preparation
(data.audit)      (dk.safe.*)    (outliers/etc)  (data.plot.*)     (data.prepare)
(data.clean)                     (data.eda)                        (data.report)
```

1. **Thin Abstraction Layer**: DataKit sits directly on top of NumPy, Pandas, Matplotlib, and Seaborn without duplicating basic data structures.
2. **Immutable Pipeline Ergonomics**: Methods that clean or transform data return a new `DataKit` instance, preserving the original data state.
3. **Strict Validation & Explained Errors**: All custom exceptions (`ColumnNotFoundError`, `ShapeMismatchError`, `ConfirmationRequiredError`) follow a three-part message structure: *what happened*, *why it matters*, and *suggested fix*.

---

## 📑 User Guide & LaTeX Documentation

Detailed offline user documentation is included in the [`docs/`](docs/) directory:
- **LaTeX Source Code**: [`docs/DataKit_User_Guide.tex`](docs/DataKit_User_Guide.tex)
- **Compiled PDF Manual**: [`docs/DataKit_User_Guide.pdf`](docs/DataKit_User_Guide.pdf)

---

## 🧪 Development & Testing

DataKit relies on `pytest` for rigorous unit testing across all 10 implementation phases.

```bash
# Run complete unit test suite (123 tests)
py -m pytest tests/ -v
```

---

## 📄 License

DataKit is licensed under the terms of the **MIT License**. See [LICENSE](LICENSE) for details.
