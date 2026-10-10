"""Error recovery hints and existing pandas schema semantics."""

import numpy as np
import pandas as pd
import pytest

from neurotabular.device import resolve_device
from neurotabular.preprocessing import UNKNOWN_CATEGORY_ID, TabularPreprocessor


def test_duplicate_columns_name_the_problem_and_fix():
    X = pd.DataFrame([[1, 2], [3, 4]], columns=["amount", "amount"])
    with pytest.raises(ValueError, match="unique.*amount.*Rename"):
        TabularPreprocessor().fit(X)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_schema_mismatch_includes_recovery_hint(change):
    X = pd.DataFrame({"amount": [1, 2], "age": [3, 4]})
    processor = TabularPreprocessor().fit(X)
    query = X.drop(columns="age") if change == "missing" else X.assign(extra=1)
    with pytest.raises(ValueError, match="schema mismatch.*same column names"):
        processor.transform(query)


def test_numeric_drift_error_names_only_offending_columns():
    processor = TabularPreprocessor().fit(pd.DataFrame({"good": [1, 2], "bad": [3, 4]}))
    with pytest.raises(ValueError) as error:
        processor.transform(pd.DataFrame({"good": ["5"], "bad": ["not_numeric"]}))
    message = str(error.value)
    assert "['bad']" in message
    assert "['good']" not in message
    assert "categorical_features" in message
    assert "refit" in message


def test_unknown_categorical_name_has_fix():
    with pytest.raises(ValueError, match="Unknown categorical_features.*existing"):
        TabularPreprocessor(categorical_features=["typo"]).fit(
            pd.DataFrame({"amount": [1, 2]})
        )


@pytest.mark.parametrize("available", [False, True])
def test_explicit_cuda_failures_keep_reason_and_cpu_recovery(monkeypatch, available):
    monkeypatch.setattr("torch.cuda.is_available", lambda: available)
    if available:
        monkeypatch.setattr("torch.cuda.device_count", lambda: 1)
        monkeypatch.setattr("torch.cuda.current_device", lambda: 0)
        monkeypatch.setattr("torch.cuda.get_device_name", lambda device: "test GPU")
        monkeypatch.setattr("torch.cuda.get_device_capability", lambda device: (7, 0))
        monkeypatch.setattr("torch.cuda.get_arch_list", lambda: ["sm_70"])
        monkeypatch.setattr(
            "neurotabular.device._run_cuda_probe",
            lambda device: "controlled probe failure",
        )
    with pytest.raises(RuntimeError) as error:
        resolve_device("cuda")
    assert "device='cpu'" in str(error.value)
    assert ("controlled probe failure" if available else "not available") in str(
        error.value
    )


@pytest.mark.parametrize(
    "dtype",
    ["string", "category", "bool", "boolean", "Int64", "UInt64", "Float32", "Float64"],
)
def test_supported_pandas_dtypes_preserve_schema_and_missing(dtype):
    categorical = dtype in {"string", "category", "bool", "boolean"}
    values = (
        [True, False, True, False]
        if dtype in {"bool", "boolean"}
        else (["a", "b", None, "a"] if categorical else [1, 2, None, 4])
    )
    X = pd.DataFrame({"column": pd.Series(values, dtype=dtype), "empty": [np.nan] * 4})
    processor = TabularPreprocessor(use_category_frequency=True).fit(X)
    original = X.copy(deep=True)
    result = processor.transform(X)
    assert ("column" in processor.categorical_features_) is categorical
    assert np.isfinite(result.numerical).all()
    pd.testing.assert_frame_equal(X, original)


@pytest.mark.parametrize("dtype", [object, "string", "category"])
def test_categorical_dtype_drift_keeps_new_levels_unknown(dtype):
    processor = TabularPreprocessor(use_category_frequency=True).fit(
        pd.DataFrame({"group": pd.Categorical(["a", "a", None])})
    )
    query = pd.DataFrame({"group": pd.Series(["new", None, "a"], dtype=dtype)})
    result = processor.transform(query)
    assert result.categorical[0, 0] == UNKNOWN_CATEGORY_ID
    assert result.numerical[0, 0] == 0
    assert result.categorical[1, 0] == 0


def test_mixed_missing_representations_and_nullable_boolean_are_supported():
    X = pd.DataFrame(
        {
            "group": pd.Series([None, np.nan, pd.NA, "a"], dtype=object),
            "active": pd.Series([True, False, None, True], dtype="boolean"),
            "number": pd.Series([None, np.nan, pd.NA, 1], dtype="Float64"),
        }
    )
    result = TabularPreprocessor().fit_transform(X)
    np.testing.assert_array_equal(result.categorical[:3, 0], [0, 0, 0])
    assert result.categorical[2, 1] == 0
    assert np.isfinite(result.numerical).all()
