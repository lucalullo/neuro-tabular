"""Product data-path equivalence, including conversion and caller ownership."""

import numpy as np
import pandas as pd
import pytest

from neurotabular.preprocessing import TabularPreprocessor


@pytest.mark.parametrize(
    "dtype",
    [
        "int32",
        "int64",
        "uint64",
        "float32",
        "float64",
        "Int64",
        "UInt64",
        "Float32",
        "Float64",
    ],
)
def test_numeric_matrix_matches_original_pandas_conversion(dtype):
    frame = pd.DataFrame(
        {
            "x": pd.Series([1, 4, 8, 12], dtype=dtype),
            "y": pd.Series([5, 9, 2, 7], dtype=dtype),
        }
    )
    original = frame.copy(deep=True)
    processor = TabularPreprocessor().fit(frame)
    reference = frame.apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan, copy=True
    )
    actual = processor._numeric_matrix(frame)
    np.testing.assert_array_equal(actual, reference)
    assert actual.strides == reference.strides
    assert not np.shares_memory(actual, frame.to_numpy(copy=False))
    processor.transform(frame)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("order", ["C", "F"])
def test_numeric_matrix_preserves_reference_layout_and_reduction_order(order):
    # Cancellation makes changing reduction order observable, beyond a check
    # that the converted individual values happen to match.
    values = np.resize([1e10, 1e-5, -1e10, 2.0], (256, 4))
    values[:, 1] = np.arange(256) * 0.01
    frame = pd.DataFrame(np.array(values, order=order))
    processor = TabularPreprocessor().fit(frame)
    reference = frame.apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan, copy=True
    )
    actual = processor._numeric_matrix(frame)
    np.testing.assert_array_equal(actual, reference)
    # pandas may choose a different owned C/F layout for an otherwise identical
    # numeric frame across supported releases; preserve the arithmetic result,
    # not an internal DataFrame-to-NumPy stride choice.
    np.testing.assert_array_equal(np.mean(actual, axis=0), np.mean(reference, axis=0))
    np.testing.assert_array_equal(np.std(actual, axis=0), np.std(reference, axis=0))


@pytest.mark.parametrize("dtype", ["object", "string", "Float64", "Int64"])
def test_fitted_numeric_feature_keeps_prediction_conversion_and_missing_semantics(
    dtype,
):
    processor = TabularPreprocessor().fit(pd.DataFrame({"x": [1, 2, 3, 4]}))
    values = ["1", None, "3", "4"] if dtype in {"object", "string"} else [1, None, 3, 4]
    query = pd.DataFrame({"x": pd.Series(values, dtype=dtype)})
    converted = query.apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan, copy=True
    )
    np.testing.assert_array_equal(processor._numeric_matrix(query), converted)
    control = pd.DataFrame(converted, columns=query.columns)
    np.testing.assert_array_equal(
        processor.transform(query).numerical, processor.transform(control).numerical
    )


def test_mixed_numeric_storage_matches_reference_and_does_not_modify_caller():
    frame = pd.DataFrame(
        {
            "integer": pd.Series([1, 2, 3, 4], dtype="Int64"),
            "single": pd.Series([1.5, np.nan, 4.0, 5.0], dtype="float32"),
            "double": [1.0, 1.5, 2.0, 3.0],
        }
    )
    original = frame.copy(deep=True)
    processor = TabularPreprocessor().fit(frame)
    reference = frame.apply(pd.to_numeric, errors="raise").to_numpy(
        dtype=np.float64, na_value=np.nan, copy=True
    )
    actual = processor._numeric_matrix(frame)
    np.testing.assert_array_equal(actual, reference)
    assert actual.strides == reference.strides
    processor.transform(frame)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize(
    "labels", ["string", "integer", "tuple", "multi", "categorical"]
)
def test_ordered_and_reordered_schema_preserve_values_and_caller_ownership(labels):
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, 3.0, 4.0],
            "y": [3.0, 7.0, 4.0, 9.0],
            "cat": ["a", "b", "a", None],
        }
    )
    if labels == "integer":
        frame.columns = [10, 20, 30]
    elif labels == "tuple":
        frame.columns = pd.Index([("x", 1), ("y", 2), ("cat", 3)], tupleize_cols=False)
    elif labels == "multi":
        frame.columns = pd.MultiIndex.from_tuples([("x", 1), ("y", 2), ("cat", 3)])
    elif labels == "categorical":
        frame.columns = pd.CategoricalIndex(frame.columns)
    original = frame.copy(deep=True)
    processor = TabularPreprocessor(use_category_frequency=True).fit(frame)
    expected = processor.transform(frame)
    reordered = processor.transform(frame.iloc[:, [2, 0, 1]])
    np.testing.assert_array_equal(expected.numerical, reordered.numerical)
    np.testing.assert_array_equal(expected.categorical, reordered.categorical)
    # Processed outputs must remain owned, even when schema selection is skipped.
    expected.numerical[0, 0] = -1234
    expected.categorical[0, 0] = 0
    pd.testing.assert_frame_equal(frame, original)
    actual = processor.transform(frame)
    np.testing.assert_array_equal(actual.numerical, reordered.numerical)
    np.testing.assert_array_equal(actual.categorical, reordered.categorical)
