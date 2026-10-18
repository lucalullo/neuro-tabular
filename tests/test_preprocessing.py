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


def test_category_frequency_is_training_only_and_handles_unknown_values():
    train = pd.DataFrame({"city": ["Rome", "Rome", "Milan", None]})
    preprocessor = TabularPreprocessor(use_category_frequency=True).fit(train)
    transformed = preprocessor.transform(
        pd.DataFrame({"city": ["Rome", "Milan", "Turin", None]})
    )
    frequency = transformed.numerical[:, 0]
    assert preprocessor.n_frequency_features_ == 1
    assert frequency[0] > frequency[1] > frequency[2]
    assert frequency[2] == 0.0
    assert frequency[3] == pytest.approx(frequency[1])


def test_research_category_cap_is_frequency_ranked_and_deterministic():
    train = pd.DataFrame(
        {
            "code": [
                "common",
                "common",
                "common",
                "second",
                "second",
                "third",
                "third",
                "rare",
            ]
        }
    )
    first = TabularPreprocessor(
        min_category_count=1,
        max_categories=2,
        use_category_frequency=True,
    ).fit(train)
    second = TabularPreprocessor(
        min_category_count=1,
        max_categories=2,
        use_category_frequency=True,
    ).fit(train)

    assert set(first.category_vocabs_["code"]) == {"common", "second"}
    assert first.rare_categories_["code"] == {"third", "rare"}
    assert first.categorical_cardinalities_ == [5]
    assert np.array_equal(
        first.transform(train).categorical,
        second.transform(train).categorical,
    )


def test_research_hash_buckets_bound_cardinality_and_keep_special_ids_distinct():
    train = pd.DataFrame({"code": ["a", "a", "b", "b", "c", "c", "d", "d", None]})
    preprocessor = TabularPreprocessor(
        min_category_count=1,
        max_categories=1,
        hash_buckets=3,
        use_category_frequency=True,
    ).fit(train)
    test = pd.DataFrame({"code": [None, "unseen", "b", "c", "d", "a"]})
    transformed = preprocessor.transform(test)
    encoded = transformed.categorical[:, 0]

    assert preprocessor.categorical_cardinalities_ == [6]
    assert encoded[0] == MISSING_CATEGORY_ID
    assert encoded[1] == UNKNOWN_CATEGORY_ID
    assert np.all((encoded[2:5] >= RARE_CATEGORY_ID) & (encoded[2:5] < 5))
    assert encoded[5] == 5
    assert np.isfinite(transformed.numerical).all()


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"max_categories": 0}, "max_categories"),
        ({"max_categories": True}, "max_categories"),
        ({"hash_buckets": -1}, "hash_buckets"),
        ({"hash_buckets": True}, "hash_buckets"),
    ],
)
def test_research_category_overflow_options_are_validated(options, message):
    with pytest.raises(ValueError, match=message):
        TabularPreprocessor(**options).fit(pd.DataFrame({"code": ["a", "b"]}))


def test_numeric_knots_are_training_only_finite_and_strictly_increasing():
    train = pd.DataFrame({"varying": [0.0, 1.0, 2.0, 3.0], "constant": [2.0] * 4})
    preprocessor = TabularPreprocessor(n_numeric_bins=6).fit(train)
    before = preprocessor.numeric_knots_.copy()
    preprocessor.transform(
        pd.DataFrame({"varying": [-1000.0, 1000.0], "constant": [2.0, 2.0]})
    )
    assert before.shape == (2, 7)
    assert np.isfinite(before).all()
    assert np.all(np.diff(before, axis=1) > 0.0)
    assert np.array_equal(before, preprocessor.numeric_knots_)


def test_schema_is_reordered_and_changes_are_rejected():
    train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    preprocessor = TabularPreprocessor().fit(train)
    expected = preprocessor.transform(train).numerical
    assert np.array_equal(preprocessor.transform(train[["b", "a"]]).numerical, expected)
    with pytest.raises(ValueError, match="missing columns"):
        preprocessor.transform(train[["a"]])
    with pytest.raises(ValueError, match="unexpected columns"):
        preprocessor.transform(train.assign(extra=1))


@pytest.mark.parametrize(
    "columns",
    [
        pd.Index(["a", "b", "c"], dtype=object),
        pd.Index(["a", "b", "c"], dtype="string"),
        pd.CategoricalIndex(["a", "b", "c"]),
        pd.Index([1, 2, 3]),
        pd.Index([7, "b", ("group", "c")], tupleize_cols=False),
        pd.MultiIndex.from_tuples([("g", "a"), ("g", "b"), ("h", "c")]),
    ],
)
def test_schema_alignment_preserves_column_index_variants(columns):
    train = pd.DataFrame([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], columns=columns)
    preprocessor = TabularPreprocessor().fit(train)
    expected = preprocessor.transform(train)
    for query in [train.copy(), train.iloc[:, ::-1]]:
        before = query.copy(deep=True)
        actual = preprocessor.transform(query)
        assert np.array_equal(actual.numerical, expected.numerical)
        assert np.array_equal(actual.categorical, expected.categorical)
        pd.testing.assert_frame_equal(query, before)


@pytest.mark.parametrize("storage", [object, "string", "category"])
def test_aligned_string_schema_keeps_missing_extra_and_duplicate_errors(storage):
    train = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
    train.columns = pd.Index(train.columns, dtype=storage)
    preprocessor = TabularPreprocessor().fit(train)
    with pytest.raises(ValueError, match="missing columns.*unexpected columns"):
        preprocessor.transform(train.rename(columns={"b": "other"}))
    duplicate = train.copy()
    duplicate.columns = ["a", "a"]
    with pytest.raises(ValueError, match="unique column names"):
        preprocessor.transform(duplicate)
    with pytest.raises(ValueError, match="at least one row"):
        preprocessor.transform(train.iloc[:0])


def test_equal_indexes_keep_null_column_name_diagnostics():
    train = pd.DataFrame(
        [[1.0, 2.0], [3.0, 4.0]],
        columns=pd.Index([float("nan"), "b"], dtype=object),
    )
    preprocessor = TabularPreprocessor().fit(train)
    query = train.copy()
    query.columns = pd.Index([float("nan"), "b"], dtype=object)
    assert query.columns.equals(train.columns)
    with pytest.raises(ValueError, match="unexpected columns"):
        preprocessor.transform(query)


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


def test_unused_pandas_categorical_levels_remain_unknown() -> None:
    X_train = pd.DataFrame(
        {
            "category": pd.Categorical(
                ["a", "a", "b"],
                categories=["a", "b", "validation_only"],
            )
        }
    )
    preprocessor = TabularPreprocessor(
        min_category_count=2, use_category_frequency=True
    )
    preprocessor.fit(X_train)

    X_validation = pd.DataFrame(
        {
            "category": pd.Categorical(
                ["validation_only"],
                categories=["a", "b", "validation_only"],
            )
        }
    )
    transformed = preprocessor.transform(X_validation)

    assert transformed.categorical[0, 0] == UNKNOWN_CATEGORY_ID
    assert transformed.numerical[0, -1] == pytest.approx(0.0)
    assert "validation_only" not in preprocessor.rare_categories_["category"]


def test_complex_numeric_values_are_rejected() -> None:
    X = pd.DataFrame({"numeric": [1 + 2j, 2 + 0j, 3 - 1j]})

    with pytest.raises(ValueError, match="real-valued.*complex"):
        TabularPreprocessor().fit(X)


def test_finite_float64_overflow_is_rejected_before_float32_conversion() -> None:
    preprocessor = TabularPreprocessor().fit(pd.DataFrame({"x": [0.0, 0.0]}))
    with pytest.raises(ValueError, match="represented safely in float32"):
        preprocessor.transform(pd.DataFrame({"x": [np.finfo(np.float64).max]}))


@pytest.mark.parametrize("storage", ["object", "string", "category"])
def test_category_frequency_preserves_training_bucket_counts(storage):
    values = ["z", "z", "z", "a", "a", "b", "c", None, None, None]
    column = pd.Series(values, dtype=storage)
    if storage == "category":
        column = column.cat.add_categories(["unused"])
    train = pd.DataFrame({"code": column})
    before = train.copy(deep=True)
    preprocessor = TabularPreprocessor(use_category_frequency=True).fit(train)
    query = pd.DataFrame({"code": ["z", "a", "b", "c", "unused", None]})

    transformed = preprocessor.transform(query)

    # Both individually rare values share their combined training frequency.
    counts = [3, 2, 2, 2, 0, 3]
    expected = np.array(
        [float(np.log1p(count) / np.log1p(len(train))) for count in counts],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(transformed.numerical[:, 0], expected)
    assert transformed.categorical[4, 0] == UNKNOWN_CATEGORY_ID
    pd.testing.assert_frame_equal(train, before)


@pytest.mark.parametrize("threshold", [1, 2, 4])
@pytest.mark.parametrize("cap", [None, 2])
@pytest.mark.parametrize("hash_buckets", [0, 3])
def test_overflow_frequency_equals_encoded_training_bucket_mass(
    threshold, cap, hash_buckets
):
    train = pd.DataFrame(
        {"code": ["z"] * 7 + ["a"] * 4 + ["b"] * 2 + ["c", "d", None, None]}
    )
    preprocessor = TabularPreprocessor(
        min_category_count=threshold,
        max_categories=cap,
        hash_buckets=hash_buckets,
        use_category_frequency=True,
    ).fit(train)
    transformed = preprocessor.transform(train)
    ids = transformed.categorical[:, 0]
    counts = np.bincount(ids, minlength=preprocessor.categorical_cardinalities_[0])
    expected = np.array(
        [float(np.log1p(int(count)) / np.log1p(len(train))) for count in counts],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(transformed.numerical[:, 0], expected[ids])
    unknown = preprocessor.transform(pd.DataFrame({"code": ["unseen"]}))
    assert unknown.categorical[0, 0] == UNKNOWN_CATEGORY_ID
    assert unknown.numerical[0, 0] == 0.0


@pytest.mark.parametrize("storage", ["object", "string", "category"])
def test_category_mapping_preserves_query_dtype_and_missing_values(storage):
    train = pd.DataFrame(
        {"code": pd.Series(["a"] * 4 + ["b"] * 3 + ["rare", None, None], dtype=storage)}
    )
    values = ["b", None, "unknown", "rare", "a", np.nan, pd.NA, "unused"]
    if storage == "category":
        train["code"] = train["code"].cat.add_categories(["unused"])
        column = pd.Series(
            pd.Categorical(values, categories=["unknown", "unused", "rare", "b", "a"])
        )
    else:
        column = pd.Series(values, dtype=storage)
    query = pd.DataFrame({"code": column})
    before = query.copy(deep=True)
    preprocessor = TabularPreprocessor(use_category_frequency=True).fit(train)
    vocabulary = preprocessor.category_vocabs_["code"].copy()

    transformed = preprocessor.transform(query)

    np.testing.assert_array_equal(
        transformed.categorical[:, 0], [4, 0, 1, 2, 3, 0, 0, 1]
    )
    counts = [3, 2, 0, 1, 4, 2, 2, 0]
    expected = np.array(
        [float(np.log1p(count) / np.log1p(len(train))) for count in counts],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(transformed.numerical[:, 0], expected)
    pd.testing.assert_frame_equal(query, before)
    assert preprocessor.category_vocabs_["code"] == vocabulary


@pytest.mark.parametrize("storage", ["object", "string", "category"])
def test_empty_frequent_vocabulary_retains_rare_and_missing_codes(storage):
    train = pd.DataFrame({"code": pd.Series(["r1", "r2", None], dtype=storage)})
    if storage == "category":
        train["code"] = train["code"].cat.add_categories(["unused"])
    preprocessor = TabularPreprocessor(use_category_frequency=True).fit(train)
    assert preprocessor.category_vocabs_["code"] == {}
    query = pd.DataFrame(
        {
            "code": pd.Series(
                ["unknown", "r1", "r2", None, np.nan, pd.NA, "unused"], dtype=storage
            )
        }
    )

    transformed = preprocessor.transform(query)

    np.testing.assert_array_equal(transformed.categorical[:, 0], [1, 2, 2, 0, 0, 0, 1])
    expected = np.array(
        [
            float(np.log1p(count) / np.log1p(len(train)))
            for count in [0, 2, 2, 1, 1, 1, 0]
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(transformed.numerical[:, 0], expected)


def test_large_nullable_integer_category_keys_remain_distinct():
    first, second, rare, unknown = [2**53 + offset for offset in [1, 3, 5, 9]]
    train = pd.DataFrame(
        {"code": pd.Series([first] * 3 + [second] * 2 + [rare, None], dtype="Int64")}
    )
    preprocessor = TabularPreprocessor(
        categorical_features=["code"], use_category_frequency=True
    ).fit(train)
    query = pd.DataFrame(
        {"code": pd.Series([second, first, unknown, None], dtype="Int64")}
    )

    transformed = preprocessor.transform(query)

    np.testing.assert_array_equal(transformed.categorical[:, 0], [4, 3, 1, 0])
    expected = np.array(
        [float(np.log1p(count) / np.log1p(len(train))) for count in [2, 3, 0, 1]],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(transformed.numerical[:, 0], expected)


@pytest.mark.parametrize("numeric_strategy", ["standard", "robust"])
@pytest.mark.parametrize("dtype", [np.float64, np.float32, np.int64])
def test_numeric_transform_preserves_shared_read_only_input(numeric_strategy, dtype):
    backing = np.arange(120, dtype=dtype).reshape(30, 4)
    if np.issubdtype(dtype, np.floating):
        backing[::3, 1] = np.nan
    backing.flags.writeable = False
    X = pd.DataFrame(backing, columns=list("abcd"), copy=False)
    original = X.copy(deep=True)
    preprocessor = TabularPreprocessor(numeric_strategy=numeric_strategy).fit(X)
    first = preprocessor.transform(X.iloc[::2, ::-1])
    expected = first.numerical.copy()
    second = preprocessor.transform(X.iloc[1::2])
    second.numerical[:] = 0.0
    pd.testing.assert_frame_equal(X, original, check_exact=True)
    np.testing.assert_array_equal(first.numerical, expected)
    assert not np.shares_memory(first.numerical, backing)
    assert not np.shares_memory(first.numerical, second.numerical)
    assert first.numerical.flags.c_contiguous


@pytest.mark.parametrize("numeric_strategy", ["standard", "robust"])
def test_nullable_numeric_transform_is_stable_across_queries_and_refit(
    numeric_strategy,
):
    X = pd.DataFrame(
        {
            "integer": pd.Series([1, None, 3, 4, 5, 6], dtype="Int64"),
            "float": pd.Series([1.0, 2.0, None, 4.0, 5.0, 6.0], dtype="Float64"),
            "category": pd.Series(["a", "a", "b", None, "b", "c"], dtype="string"),
        }
    )
    original = X.copy(deep=True)
    preprocessor = TabularPreprocessor(
        numeric_strategy=numeric_strategy, use_category_frequency=True
    ).fit(X)
    expected = preprocessor.transform(X).numerical.copy()
    preprocessor.transform(X.iloc[::-1, ::-1])
    np.testing.assert_array_equal(preprocessor.transform(X).numerical, expected)
    preprocessor.fit(X.iloc[::-1])
    refitted = preprocessor.transform(X)
    fresh = TabularPreprocessor(
        numeric_strategy=numeric_strategy, use_category_frequency=True
    ).fit_transform(X.iloc[::-1])
    np.testing.assert_array_equal(refitted.numerical, fresh.numerical[::-1])
    np.testing.assert_array_equal(refitted.categorical, fresh.categorical[::-1])
    pd.testing.assert_frame_equal(X, original, check_exact=True)
