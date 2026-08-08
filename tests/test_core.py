"""Tests for DataKit core: instantiation, properties, copy semantics, type hierarchy."""
import pytest
import pandas as pd
from pathlib import Path

from datakit import DataKit, config, __version__
from datakit.core.exceptions import (
    DataKitError,
    ColumnNotFoundError,
    IncompatibleColumnTypeError,
    EmptyDataError,
    InsufficientDataError,
    ConfirmationRequiredError,
    ShapeMismatchError,
    ConfigError,
)
from datakit.core.warnings import (
    DataKitWarning,
    ImplicitBroadcastWarning,
    IndexAlignmentWarning,
    LargeDatasetWarning,
    DataLossWarning,
    HighCardinalityWarning,
    ConstantColumnWarning,
    PotentialLeakageWarning,
    HighCardinalityEncodingWarning,
)


# ---- Version ----

class TestVersion:
    def test_version_exists(self):
        assert __version__ is not None
        assert isinstance(__version__, str)

    def test_version_format(self):
        parts = __version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)


# ---- DataKit Instantiation ----

class TestDataKitInit:
    def test_from_dataframe(self, sample_df):
        data = DataKit(sample_df)
        assert isinstance(data.df, pd.DataFrame)
        assert data.df.shape == sample_df.shape

    def test_dk_read_function(self, insurance_csv_path):
        import datakit as dk
        data = dk.read(str(insurance_csv_path))
        assert isinstance(data, DataKit)
        assert data.df.shape[0] > 0

    def test_datakit_read_classmethod(self, insurance_csv_path):
        data = DataKit.read(insurance_csv_path)
        assert isinstance(data, DataKit)
        assert data.df.shape[0] > 0

    def test_from_dict(self):
        data = DataKit({"a": [1, 2, 3], "b": [4, 5, 6]})
        assert data.df.shape == (3, 2)
        assert list(data.df.columns) == ["a", "b"]

    def test_from_csv(self, insurance_csv_path):
        data = DataKit(str(insurance_csv_path))
        assert isinstance(data.df, pd.DataFrame)
        assert data.df.shape[0] > 0

    def test_from_json(self, tmp_path):
        json_path = tmp_path / "data.json"
        df_orig = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        df_orig.to_json(json_path)
        data = DataKit(str(json_path))
        assert isinstance(data.df, pd.DataFrame)
        assert data.df.shape == (2, 2)

    def test_from_parquet_or_missing_dependency(self, tmp_path):
        parquet_path = tmp_path / "data.parquet"
        df_orig = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        try:
            df_orig.to_parquet(parquet_path)
            data = DataKit(str(parquet_path))
            assert isinstance(data.df, pd.DataFrame)
            assert data.df.shape == (2, 2)
        except ImportError:
            # If pyarrow/fastparquet not installed in environment, verify clear ImportError raised
            with pytest.raises(ImportError, match="required to read Parquet"):
                DataKit(str(parquet_path))

    def test_from_excel_or_missing_dependency(self, tmp_path):
        excel_path = tmp_path / "data.xlsx"
        df_orig = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        try:
            df_orig.to_excel(excel_path, index=False)
            data = DataKit(str(excel_path))
            assert isinstance(data.df, pd.DataFrame)
            assert data.df.shape == (2, 2)
        except ImportError:
            with pytest.raises(ImportError, match="openpyxl is required"):
                DataKit(str(excel_path))

    def test_from_path_object(self, insurance_csv_path):
        data = DataKit(insurance_csv_path)
        assert isinstance(data.df, pd.DataFrame)

    def test_invalid_source_type(self):
        with pytest.raises(TypeError, match="DataKit expects"):
            DataKit(42)

    def test_unsupported_file_format(self, tmp_path):
        fake = tmp_path / "data.xyz"
        fake.write_text("hello")
        with pytest.raises(ValueError, match="Unsupported file format"):
            DataKit(str(fake))


# ---- Copy Semantics ----

class TestCopySemantics:
    def test_dataframe_is_copied(self, sample_df):
        """Mutating the original DataFrame must NOT affect DataKit's internal copy."""
        data = DataKit(sample_df)
        original_value = data.df.iloc[0, 0]
        sample_df.iloc[0, 0] = -9999
        assert data.df.iloc[0, 0] == original_value

    def test_df_property_returns_live_reference(self, sample_datakit):
        """data.df should return the actual internal DataFrame, not a copy of it."""
        df1 = sample_datakit.df
        df2 = sample_datakit.df
        assert df1 is df2


# ---- Display ----

class TestDisplay:
    def test_repr(self, sample_datakit):
        r = repr(sample_datakit)
        assert "DataKit" in r
        assert "rows=" in r
        assert "columns=" in r

    def test_repr_html(self, sample_datakit):
        html = sample_datakit._repr_html_()
        assert "DataKit" in html
        assert "<" in html  # It's HTML


# ---- Exception Hierarchy ----

class TestExceptionHierarchy:
    def test_all_inherit_from_datakit_error(self):
        """User story #24: all exceptions inherit from common base."""
        exceptions = [
            ColumnNotFoundError, IncompatibleColumnTypeError,
            EmptyDataError, InsufficientDataError,
            ConfirmationRequiredError, ShapeMismatchError, ConfigError,
        ]
        for exc_class in exceptions:
            assert issubclass(exc_class, DataKitError)
            assert issubclass(exc_class, Exception)

    def test_column_not_found_suggests_close_match(self):
        """User story #17: column-not-found errors suggest closest name."""
        err = ColumnNotFoundError("agee", ["age", "salary", "department"])
        msg = str(err)
        assert "agee" in msg
        assert "age" in msg  # Should suggest 'age' as closest match

    def test_column_not_found_no_close_match(self):
        err = ColumnNotFoundError("zzzzz", ["age", "salary"])
        msg = str(err)
        assert "zzzzz" in msg

    def test_incompatible_column_type_message(self):
        err = IncompatibleColumnTypeError("name", "object", ["int64", "float64"])
        msg = str(err)
        assert "name" in msg
        assert "object" in msg

    def test_confirmation_required_message(self):
        err = ConfirmationRequiredError("clean(missing='drop')")
        msg = str(err)
        assert "confirm" in msg.lower()

    def test_shape_mismatch_includes_shapes(self):
        err = ShapeMismatchError((5,), (5, 1), (5, 5))
        msg = str(err)
        assert "(5,)" in msg
        assert "(5, 1)" in msg

    def test_can_catch_all_with_base(self):
        """User story #24 acceptance criterion."""
        with pytest.raises(DataKitError):
            raise ColumnNotFoundError("x", ["a", "b"])


# ---- Warning Hierarchy ----

class TestWarningHierarchy:
    def test_all_inherit_from_datakit_warning(self):
        warnings_list = [
            ImplicitBroadcastWarning, IndexAlignmentWarning,
            LargeDatasetWarning, DataLossWarning,
            HighCardinalityWarning, ConstantColumnWarning,
            PotentialLeakageWarning, HighCardinalityEncodingWarning,
        ]
        for warn_class in warnings_list:
            assert issubclass(warn_class, DataKitWarning)
            assert issubclass(warn_class, UserWarning)


# ---- Config ----

class TestConfig:
    @pytest.fixture(autouse=True)
    def reset_config(self):
        yield
        config.reset()

    def test_get_default(self):
        assert config.get("figsize") == (10, 6)
        assert config.get("dpi") == 100

    def test_set_and_get(self):
        config.set(dpi=300)
        assert config.get("dpi") == 300

    def test_set_unknown_key_raises(self):
        with pytest.raises(ConfigError):
            config.set(nonexistent_key=42)

    def test_get_unknown_key_raises(self):
        with pytest.raises(ConfigError):
            config.get("nonexistent_key")

    def test_reset(self):
        config.set(dpi=999)
        config.reset()
        assert config.get("dpi") == 100

    def test_threshold_defaults_exist(self):
        """Ensure all thresholds needed by later phases are present from Phase 0."""
        expected_keys = [
            "missing_critical_threshold",
            "missing_warning_threshold",
            "high_cardinality_threshold",
            "near_constant_threshold",
            "leakage_correlation_threshold",
            "id_like_uniqueness_threshold",
            "data_loss_warning_threshold",
            "large_dataset_mb",
            "outlier_iqr_multiplier",
            "outlier_zscore_threshold",
            "figsize",
            "dpi",
            "theme",
        ]
        for key in expected_keys:
            val = config.get(key)
            assert val is not None, f"Config key '{key}' should have a default"


# ---- Insurance Fixture Sanity ----

class TestInsuranceFixture:
    def test_has_expected_columns(self, insurance_datakit):
        expected = {"age", "sex", "bmi", "children", "smoker", "region", "charges", "plan_type", "id"}
        assert expected.issubset(set(insurance_datakit.df.columns))

    def test_has_missing_values(self, insurance_datakit):
        assert insurance_datakit.df.isnull().any().any(), "Fixture should have missing values"

    def test_has_duplicates(self, insurance_datakit):
        assert insurance_datakit.df.duplicated().any(), "Fixture should have duplicate rows"

    def test_has_near_constant_column(self, insurance_datakit):
        plan_counts = insurance_datakit.df["plan_type"].value_counts()
        most_common_ratio = plan_counts.iloc[0] / len(insurance_datakit.df)
        assert most_common_ratio >= 0.9, "plan_type should be near-constant"
