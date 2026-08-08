import difflib

class DataKitError(Exception):
    """Base class for all DataKit exceptions."""
    pass

class ColumnNotFoundError(DataKitError):
    """Raised when a specified column is not found in the DataFrame."""
    def __init__(self, column: str, available_columns: list[str]):
        matches = difflib.get_close_matches(column, available_columns, n=1, cutoff=0.6)
        suggestion = f" Did you mean '{matches[0]}'?" if matches else ""
        msg = f"'{column}' not found in DataFrame.{suggestion} Available columns: {available_columns}"
        super().__init__(msg)

class IncompatibleColumnTypeError(DataKitError):
    """Raised when a column has a type incompatible with the operation."""
    def __init__(self, column: str, actual_dtype: str, expected_dtypes: list[str]):
        msg = f"Column '{column}' has incompatible type '{actual_dtype}'. Expected types: {expected_dtypes}. Ensure the column is converted to an expected type before this operation."
        super().__init__(msg)

class EmptyDataError(DataKitError):
    """Raised when a DataFrame has 0 rows but data is required."""
    def __init__(self):
        super().__init__("The DataFrame has 0 rows. This operation requires a non-empty DataFrame. Check upstream steps that may have filtered out all data.")

class InsufficientDataError(DataKitError):
    """Raised when there is not enough data to perform the operation."""
    def __init__(self, message: str = "Not enough data for operation. Provide a larger dataset to proceed."):
        super().__init__(message)

class ConfirmationRequiredError(DataKitError):
    """Raised when a destructive operation is called without confirmation."""
    def __init__(self, operation: str):
        super().__init__(f"Operation '{operation}' requires confirmation because it is destructive or has high impact. Pass 'confirm=True' to proceed.")

class ShapeMismatchError(DataKitError):
    """Raised when data structures have mismatched shapes."""
    def __init__(self, shape_a: tuple, shape_b: tuple, result_shape: tuple):
        super().__init__(f"Shape mismatch: {shape_a} and {shape_b} cannot be operated together or resulted in unexpected shape {result_shape}. Align the dimensions properly.")

class ConfigError(DataKitError):
    """Raised when an invalid configuration value is encountered."""
    def __init__(self, message: str):
        super().__init__(message)
