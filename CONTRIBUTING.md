# Contributing to DataKit

Thank you for your interest in contributing to DataKit!

DataKit is built around strict architectural principles defined in its Product Requirement Document (PRD). All pull requests and contributions are evaluated against these criteria.

---

## 🏛 Core Architectural Principles (PRD §4 & §32)

1. **Explicit Safety First**: Never silently drop rows or mutate data. Operations that modify data must be non-destructive (return a copy) and require explicit confirmation (`confirm=True`) for destructive transforms.
2. **No Silent State Mutations**: Methods on `DataKit` never mutate internal DataFrame state in-place.
3. **100% Test Coverage Gate (PRD §24)**: Every new function or modification must include unit tests covering both nominal cases and edge cases (§29 acceptance criteria). 100% of existing unit tests must pass cleanly.
4. **Escape Hatch Requirement (PRD §10)**: All abstractions must provide clean access to underlying objects (`.df`, `.fig`, `.ax`, `ax=`).

---

## 📜 Docstring Standard (PRD §7)

Every public function across `src/datakit/` must follow the standardized 8-part docstring template:

```python
def my_function(df: pd.DataFrame, param: int = 10) -> OutputResult:
    """Short summary of the function.

    Purpose:
        Detailed explanation of why this function exists and what problem it solves.

    Params:
        df (pd.DataFrame): Input DataFrame.
        param (int): Description and default value.

    Returns:
        OutputResult: Structured result object or DataFrame.

    Mutates: No (returns copy / new object).
    Chainable: No.
    Version Added: v0.1.0

    Errors:
        DataKitError: Raised under specific failure conditions.

    Warnings:
        DataKitWarning: Emitted under specific potential risk conditions.
    """
```

---

## 🧪 Testing Guidelines

Run the full unit test suite locally before submitting a pull request:

```bash
# Run pytest test suite
py -m pytest tests/ -v

# Run static type checking with mypy
py -m mypy src/datakit --ignore-missing-imports
```

Pull requests will not be merged if any unit test fails or `mypy` reports type errors.

---

## 🛠 Proposing New Public Functions

New public functions must be explicitly evaluated against PRD principles before implementation. Before opening a PR for a new function, open an issue detailing:
1. What problem it solves.
2. Why existing APIs (`.df`, `audit()`, `clean()`, etc.) do not already cover it.
3. Which PRD principle (§4) it aligns with.
