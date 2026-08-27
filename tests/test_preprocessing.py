import numpy as np
import pandas as pd
import pytest

from neurotabular.preprocessing import (
    MISSING_CATEGORY_ID,
    RARE_CATEGORY_ID,
    UNKNOWN_CATEGORY_ID,
    TabularPreprocessor,
)


def test_robust_numeric_preprocessing_is_training_only_and_keeps_missingness():
    train = pd.DataFrame({"amount": [1.0, 2.0, np.nan, 5.0]})
    preprocessor = TabularPreprocessor(numeric_strategy="robust").fit(train)
    before = (
        preprocessor.numeric_medians_.copy(),
        preprocessor.numeric_centers_.copy(),
        preprocessor.numeric_scales_.copy(),
    )

    transformed = preprocessor.transform(
        pd.DataFrame({"amount": [1000.0, np.nan]})
    ).numerical

    assert transformed.dtype == np.float32
    assert transformed.shape == (2, 2)
    assert np.array_equal(transformed[:, 1], np.array([0.0, 1.0]))
    assert np.max(np.abs(transformed[:, 0])) <= 5.0
    assert all(
        np.array_equal(left, right)
        for left, right in zip(
            before,
            (
                preprocessor.numeric_medians_,
                preprocessor.numeric_centers_,
                preprocessor.numeric_scales_,
            ),
        )
    )


@pytest.mark.parametrize("numeric_strategy", ["standard", "robust"])
def test_numeric_strategies_handle_constant_and_all_missing(numeric_strategy):
    X = pd.DataFrame({"constant": [3.0] * 5, "empty": [np.nan] * 5})
    preprocessor = TabularPreprocessor(numeric_strategy=numeric_strategy).fit(X)
    transformed = preprocessor.transform(X).numerical
    assert transformed.shape == (5, 4)
    assert np.isfinite(transformed).all()
    assert preprocessor.numeric_medians_[1] == 0.0
    assert preprocessor.numeric_scales_[0] == 1.0


def test_categorical_autodetection_and_explicit_integer_feature_are_additive():
    X = pd.DataFrame(
        {
            "object": pd.Series(["a", "b", "a"], dtype=object),
            "string": pd.Series(["a", "b", "a"], dtype="string"),
            "category": pd.Series(["a", "b", "a"], dtype="category"),
            "boolean": pd.Series([True, False, True], dtype=bool),
            "postal_code": [10, 20, 10],
            "value": [1.0, 2.0, 3.0],
        }
    )
    preprocessor = TabularPreprocessor(categorical_features=["postal_code"]).fit(X)
    assert preprocessor.categorical_features_ == [
        "object",
        "string",
        "category",
        "boolean",
        "postal_code",
    ]
    assert preprocessor.numeric_features_ == ["value"]


def test_missing_unknown_rare_and_frequent_categories_have_distinct_ids():
    train = pd.DataFrame({"city": ["Rome", "Rome", "Milan", None]})
    preprocessor = TabularPreprocessor(min_category_count=2).fit(train)
    test = pd.DataFrame({"city": [None, "Turin", "Milan", "Rome"]})
    encoded = preprocessor.transform(test).categorical[:, 0]
    assert np.array_equal(
        encoded,
        [MISSING_CATEGORY_ID, UNKNOWN_CATEGORY_ID, RARE_CATEGORY_ID, 3],
    )


def test_all_missing_categorical_column_is_safe():
    X = pd.DataFrame({"empty": pd.Series([None] * 5, dtype=object)})
    preprocessor = TabularPreprocessor().fit(X)
    encoded = preprocessor.transform(X).categorical
    assert np.array_equal(encoded, np.zeros((5, 1), dtype=np.int64))
    assert preprocessor.categorical_cardinalities_ == [3]


def test_schema_is_reordered_and_changes_are_rejected():
    train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    preprocessor = TabularPreprocessor().fit(train)
    expected = preprocessor.transform(train).numerical
    assert np.array_equal(preprocessor.transform(train[["b", "a"]]).numerical, expected)
    with pytest.raises(ValueError, match="missing columns"):
        preprocessor.transform(train[["a"]])
    with pytest.raises(ValueError, match="unexpected columns"):
        preprocessor.transform(train.assign(extra=1))


def test_infinity_is_rejected_during_fit_and_transform():
    with pytest.raises(ValueError, match="infinite values"):
        TabularPreprocessor().fit(pd.DataFrame({"x": [1.0, np.inf]}))
    preprocessor = TabularPreprocessor().fit(pd.DataFrame({"x": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="infinite values"):
        preprocessor.transform(pd.DataFrame({"x": [1.0, -np.inf]}))


@pytest.mark.parametrize(
    ("categorical_features", "message"),
    [
        (["missing"], "Unknown categorical_features"),
        (["code", "code"], "duplicate column names"),
        ("code", "iterable of column names"),
    ],
)
def test_invalid_explicit_categorical_features(categorical_features, message):
    X = pd.DataFrame({"code": [1, 2, 3]})
    with pytest.raises((TypeError, ValueError), match=message):
        TabularPreprocessor(categorical_features=categorical_features).fit(X)


def test_transform_before_fit_and_invalid_frames_are_clear():
    with pytest.raises(RuntimeError, match="fitted"):
        TabularPreprocessor().transform(pd.DataFrame({"x": [1]}))
    with pytest.raises(TypeError, match="DataFrame"):
        TabularPreprocessor().fit(np.ones((2, 2)))
    with pytest.raises(ValueError, match="unique"):
        TabularPreprocessor().fit(pd.DataFrame(np.ones((2, 2)), columns=["x", "x"]))
