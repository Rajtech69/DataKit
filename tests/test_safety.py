"""Tests for dk.safe.* safety operations."""
import warnings
import numpy as np
import pandas as pd
import pytest

import datakit as dk
from datakit.core.exceptions import ShapeMismatchError
from datakit.core.results import AlignCheckResult, ShapeCheckResult
from datakit.core.warnings import ImplicitBroadcastWarning, IndexAlignmentWarning


class TestSafeArithmetic:
    def test_section_29_subtract_acceptance_criteria(self):
        """PRD §29 Acceptance Criteria for dk.safe.subtract():

        Given two arrays of shape (n,) and (n,1), when dk.safe.subtract(a, b) is called,
        then an ImplicitBroadcastWarning (or ShapeMismatchError in strict mode) is raised
        naming both shapes and the resulting shape, before returning a result — reproducing
        and flagging exactly the 'hallucinated outer product' scenario documented in the source.
        """
        n = 5
        a = np.arange(n)          # shape (5,)
        b = np.arange(n)[:, None]  # shape (5, 1)

        # Non-strict mode: warns
        with pytest.warns(ImplicitBroadcastWarning) as record:
            res = dk.safe.subtract(a, b)

        assert res.shape == (n, n)
        warning_msg = str(record[0].message)
        assert "(5,)" in warning_msg
        assert "(5, 1)" in warning_msg
        assert "(5, 5)" in warning_msg

        # Strict mode: raises ShapeMismatchError
        with pytest.raises(ShapeMismatchError) as exc_info:
            dk.safe.subtract(a, b, strict=True)

        err_msg = str(exc_info.value)
        assert "(5,)" in err_msg
        assert "(5, 1)" in err_msg
        assert "(5, 5)" in err_msg

    def test_same_rank_no_warning(self):
        a = np.array([1, 2, 3])
        b = np.array([4, 5, 6])
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            res = dk.safe.subtract(a, b)
        np.testing.assert_array_equal(res, [-3, -3, -3])

    def test_safe_add_multiply_divide(self):
        a = np.array([10, 20])
        b = np.array([[1], [2]])  # (2,1) vs (2,)

        with pytest.warns(ImplicitBroadcastWarning):
            res_add = dk.safe.add(a, b)
            assert res_add.shape == (2, 2)

        with pytest.warns(ImplicitBroadcastWarning):
            res_mul = dk.safe.multiply(a, b)
            assert res_mul.shape == (2, 2)

        with pytest.warns(ImplicitBroadcastWarning):
            res_div = dk.safe.divide(a, b)
            assert res_div.shape == (2, 2)


class TestCheckShapes:
    def test_check_shapes_implicit_broadcast(self):
        a = np.ones((5,))
        b = np.ones((5, 1))
        res = dk.safe.check_shapes(a, b)
        assert isinstance(res, ShapeCheckResult)
        assert res.compatible is True
        assert res.is_implicit_broadcast is True
        assert res.result_shape == (5, 5)

    def test_check_shapes_equal_rank(self):
        a = np.ones((5, 2))
        b = np.ones((5, 2))
        res = dk.safe.check_shapes(a, b)
        assert res.compatible is True
        assert res.is_implicit_broadcast is False
        assert res.result_shape == (5, 2)

    def test_check_shapes_incompatible(self):
        a = np.ones((5,))
        b = np.ones((4,))
        res = dk.safe.check_shapes(a, b)
        assert res.compatible is False
        assert res.result_shape is None


class TestIsView:
    def test_is_view_detection(self):
        arr = np.arange(10)
        slice_arr = arr[1:5]
        copy_arr = arr[1:5].copy()

        assert dk.safe.is_view(slice_arr) is True
        assert dk.safe.is_view(copy_arr) is False


class TestAlignCheck:
    def test_align_check_matching_indices(self):
        df1 = pd.DataFrame({"a": [1, 2]}, index=[0, 1])
        df2 = pd.DataFrame({"b": [3, 4]}, index=[0, 1])

        res = dk.safe.align_check(df1, df2)
        assert isinstance(res, AlignCheckResult)
        assert res.match_pct == 100.0
        assert len(res.non_overlapping_df1) == 0

    def test_align_check_mismatched_indices_warns(self):
        df1 = pd.DataFrame({"a": [1, 2]}, index=[0, 1])
        df2 = pd.DataFrame({"b": [3, 4]}, index=[1, 2])

        with pytest.warns(IndexAlignmentWarning, match="Index sets do not match"):
            res = dk.safe.align_check(df1, df2)

        assert res.match_pct < 100.0
        assert 0 in res.non_overlapping_df1
        assert 2 in res.non_overlapping_df2


class TestReshapeColumn:
    def test_reshape_column(self):
        import numpy as np
        s = pd.Series([1, 2, 3, 4, 5])
        col_vec = dk.safe.reshape_column(s)
        assert isinstance(col_vec, np.ndarray)
        assert col_vec.shape == (5, 1)
