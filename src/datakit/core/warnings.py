class DataKitWarning(UserWarning):
    """Base class for all DataKit warnings."""
    pass

class ImplicitBroadcastWarning(DataKitWarning):
    """Warning for implicit broadcasting."""
    pass

class IndexAlignmentWarning(DataKitWarning):
    """Warning for index alignment issues."""
    pass

class LargeDatasetWarning(DataKitWarning):
    """Warning when dataset is large."""
    pass

class DataLossWarning(DataKitWarning):
    """Warning for potential data loss."""
    pass

class HighCardinalityWarning(DataKitWarning):
    """Warning for high cardinality categorical data."""
    pass

class ConstantColumnWarning(DataKitWarning):
    """Warning for columns with near constant values."""
    pass

class PotentialLeakageWarning(DataKitWarning):
    """Warning for potential data leakage."""
    pass

class HighCardinalityEncodingWarning(DataKitWarning):
    """Warning for encoding high cardinality features."""
    pass
