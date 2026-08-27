"""Leakage-safe preprocessing for heterogeneous pandas DataFrames."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Integral
from time import perf_counter
from typing import Hashable

import numpy as np
import pandas as pd
from pandas.api.types import is_bool_dtype, is_object_dtype, is_string_dtype

MISSING_CATEGORY_ID = 0
UNKNOWN_CATEGORY_ID = 1
RARE_CATEGORY_ID = 2


@dataclass(frozen=True)
class ProcessedTable:
    """Contiguous numerical and categorical arrays ready for PyTorch."""

    numerical: np.ndarray
    categorical: np.ndarray

    @property
    def n_samples(self) -> int:
        """Return the number of rows."""

        return self.numerical.shape[0]

    @property
    def nbytes(self) -> int:
        """Return the combined array storage in bytes."""

        return self.numerical.nbytes + self.categorical.nbytes


class TabularPreprocessor:
    """Learn numerical statistics and categorical vocabularies.

    Numerical values are median-imputed, scaled, and paired with missing
    indicators. The optional robust strategy adds robust scaling and smooth
    clipping. Categorical IDs reserve ``0`` for missing values, ``1`` for
    unseen values, and ``2`` for rare values seen during fitting.
    """

    def __init__(
        self,
        categorical_features: Iterable[Hashable] | None = None,
        min_category_count: int = 2,
        *,
        numeric_strategy: str = "standard",
        smooth_clip_limit: float = 5.0,
    ) -> None:
        self.categorical_features = categorical_features
        self.min_category_count = min_category_count
        self.numeric_strategy = numeric_strategy
        self.smooth_clip_limit = smooth_clip_limit

    def fit(self, X: pd.DataFrame) -> TabularPreprocessor:
        """Fit all preprocessing state from training rows only."""

        started = perf_counter()
        self._validate_options()
        X = self._validate_dataframe(X, fitting=True)
        self.feature_names_ = list(X.columns)
        explicit = set(self._validate_explicit_categoricals(X))
        self.categorical_features_ = [
            column
            for column in self.feature_names_
            if column in explicit or self._is_categorical_dtype(X[column].dtype)
        ]
        categorical_set = set(self.categorical_features_)
        self.numeric_features_ = [
            column for column in self.feature_names_ if column not in categorical_set
        ]
        schema_finished = perf_counter()

        numeric_started = perf_counter()
        self._fit_numeric(X)
        numeric_finished = perf_counter()

        self.category_vocabs_: dict[Hashable, dict[object, int]] = {}
        self.rare_categories_: dict[Hashable, set[object]] = {}
        self.categorical_cardinalities_: list[int] = []
        for column in self.categorical_features_:
            series = X[column]
            try:
                counts = series.value_counts(dropna=True, sort=False)
                frequent = [
                    value
                    for value, count in counts.items()
                    if int(count) >= self.min_category_count
                ]
                rare = {
                    value
                    for value, count in counts.items()
                    if int(count) < self.min_category_count
                }
                vocabulary = {value: index + 3 for index, value in enumerate(frequent)}
            except TypeError as exc:
                raise ValueError(
                    f"Categorical column {column!r} contains unhashable values."
                ) from exc
            self.category_vocabs_[column] = vocabulary
            self.rare_categories_[column] = rare
            self.categorical_cardinalities_.append(len(vocabulary) + 3)
        categorical_finished = perf_counter()

        self.n_numeric_outputs_ = 2 * len(self.numeric_features_)
        self.is_fitted_ = True
        self.fit_profile_ = {
            "schema_seconds": schema_finished - started,
            "numeric_statistics_seconds": numeric_finished - numeric_started,
            "categorical_vocabulary_seconds": categorical_finished - numeric_finished,
            "total_seconds": categorical_finished - started,
        }
        return self

    def transform(self, X: pd.DataFrame) -> ProcessedTable:
        """Transform rows with the state learned by :meth:`fit`."""

        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("TabularPreprocessor must be fitted before transform().")
        started = perf_counter()
        X = self._validate_dataframe(X, fitting=False)
        self._validate_schema(X)
        X = X.loc[:, self.feature_names_]
        schema_finished = perf_counter()

        numerical = self._transform_numeric(X)
        numeric_finished = perf_counter()
        categorical = self._transform_categorical(X)
        categorical_finished = perf_counter()
        result = ProcessedTable(
            numerical=np.ascontiguousarray(numerical, dtype=np.float32),
            categorical=np.ascontiguousarray(categorical, dtype=np.int64),
        )
        conversion_finished = perf_counter()
        self.last_transform_profile_ = {
            "schema_seconds": schema_finished - started,
            "numeric_transform_seconds": numeric_finished - schema_finished,
            "categorical_encoding_seconds": categorical_finished - numeric_finished,
            "numpy_conversion_seconds": conversion_finished - categorical_finished,
            "total_seconds": conversion_finished - started,
        }
        return result

    def fit_transform(self, X: pd.DataFrame) -> ProcessedTable:
        """Fit on ``X`` and transform it."""

        return self.fit(X).transform(X)

    def _fit_numeric(self, X: pd.DataFrame) -> None:
        if not self.numeric_features_:
            self.numeric_medians_ = np.empty(0, dtype=np.float64)
            self.numeric_centers_ = np.empty(0, dtype=np.float64)
            self.numeric_scales_ = np.empty(0, dtype=np.float64)
            return
        values = self._numeric_matrix(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            medians = np.nanmedian(values, axis=0)
        medians = np.where(np.isfinite(medians), medians, 0.0)
        imputed = np.where(np.isnan(values), medians, values)
        if self.numeric_strategy == "standard":
            centers = np.mean(imputed, axis=0, dtype=np.float64)
            scales = np.std(imputed, axis=0, dtype=np.float64)
        else:
            centers = medians.copy()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                quartiles = np.nanpercentile(values, [25.0, 75.0], axis=0)
            scales = (quartiles[1] - quartiles[0]) / 1.349
        invalid_scale = ~np.isfinite(scales) | (np.abs(scales) < 1e-12)
        scales = np.where(invalid_scale, 1.0, scales)
        if not np.isfinite(centers).all():
            raise ValueError("Numerical features produce non-finite center statistics.")
        self.numeric_medians_ = medians.astype(np.float64, copy=False)
        self.numeric_centers_ = centers.astype(np.float64, copy=False)
        self.numeric_scales_ = scales.astype(np.float64, copy=False)

    def _transform_numeric(self, X: pd.DataFrame) -> np.ndarray:
        n_samples = len(X)
        if not self.numeric_features_:
            return np.empty((n_samples, 0), dtype=np.float32)
        values = self._numeric_matrix(X)
        missing = np.isnan(values)
        imputed = np.where(missing, self.numeric_medians_, values)
        with np.errstate(over="ignore", invalid="ignore"):
            scaled = (imputed - self.numeric_centers_) / self.numeric_scales_
            if self.numeric_strategy == "robust":
                limit = self.smooth_clip_limit
                scaled = limit * np.tanh(scaled / limit)
        if not np.isfinite(scaled).all():
            raise ValueError("Numerical preprocessing produced non-finite values.")
        return np.concatenate(
            (scaled.astype(np.float32), missing.astype(np.float32)), axis=1
        )

    def _transform_categorical(self, X: pd.DataFrame) -> np.ndarray:
        n_samples = len(X)
        if not self.categorical_features_:
            return np.empty((n_samples, 0), dtype=np.int64)
        categorical = np.empty(
            (n_samples, len(self.categorical_features_)), dtype=np.int64
        )
        for index, column in enumerate(self.categorical_features_):
            series = X[column]
            missing = series.isna().to_numpy(dtype=bool)
            try:
                mapped = series.map(self.category_vocabs_[column])
                encoded = mapped.to_numpy(dtype=np.float64, na_value=np.nan)
                encoded = np.where(np.isnan(encoded), UNKNOWN_CATEGORY_ID, encoded)
                rare = self.rare_categories_[column]
                if rare:
                    encoded[series.isin(rare).to_numpy(dtype=bool)] = RARE_CATEGORY_ID
            except TypeError as exc:
                raise ValueError(
                    f"Categorical column {column!r} contains unhashable values."
                ) from exc
            encoded[missing] = MISSING_CATEGORY_ID
            categorical[:, index] = encoded.astype(np.int64, copy=False)
        return categorical

    def _numeric_matrix(self, X: pd.DataFrame) -> np.ndarray:
        try:
            converted = X.loc[:, self.numeric_features_].apply(
                pd.to_numeric, errors="raise"
            )
            values = converted.to_numpy(dtype=np.float64, na_value=np.nan, copy=True)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "A feature treated as numerical contains non-numeric values; add "
                "the column to categorical_features if appropriate."
            ) from exc
        if np.isinf(values).any():
            flags = np.isinf(values).any(axis=0)
            bad_columns = [
                column
                for column, has_inf in zip(self.numeric_features_, flags, strict=True)
                if has_inf
            ]
            raise ValueError(
                f"Numerical columns contain infinite values: {bad_columns!r}."
            )
        return values

    def _validate_schema(self, X: pd.DataFrame) -> None:
        missing = [column for column in self.feature_names_ if column not in X.columns]
        extra = [column for column in X.columns if column not in self.feature_names_]
        if not missing and not extra:
            return
        details: list[str] = []
        if missing:
            details.append(f"missing columns: {missing!r}")
        if extra:
            details.append(f"unexpected columns: {extra!r}")
        raise ValueError("Input schema mismatch (" + "; ".join(details) + ").")

    def _validate_explicit_categoricals(self, X: pd.DataFrame) -> list[Hashable]:
        if self.categorical_features is None:
            return []
        if isinstance(self.categorical_features, (str, bytes)):
            raise TypeError("categorical_features must be an iterable of column names.")
        try:
            explicit = list(self.categorical_features)
        except TypeError as exc:
            raise TypeError(
                "categorical_features must be an iterable of column names."
            ) from exc
        duplicates = [
            name for index, name in enumerate(explicit) if name in explicit[:index]
        ]
        if duplicates:
            raise ValueError(
                f"categorical_features contains duplicate column names: {duplicates!r}."
            )
        unknown = [name for name in explicit if name not in X.columns]
        if unknown:
            raise ValueError(f"Unknown categorical_features: {unknown!r}.")
        return explicit

    def _validate_options(self) -> None:
        if (
            not isinstance(self.min_category_count, Integral)
            or isinstance(self.min_category_count, bool)
            or self.min_category_count < 1
        ):
            raise ValueError("min_category_count must be a positive integer.")
        if self.numeric_strategy not in {"standard", "robust"}:
            raise ValueError("numeric_strategy must be 'standard' or 'robust'.")
        if self.smooth_clip_limit <= 0.0:
            raise ValueError("smooth_clip_limit must be positive.")

    @staticmethod
    def _is_categorical_dtype(dtype: object) -> bool:
        return bool(
            is_object_dtype(dtype)
            or is_string_dtype(dtype)
            or isinstance(dtype, pd.CategoricalDtype)
            or is_bool_dtype(dtype)
        )

    @staticmethod
    def _validate_dataframe(X: pd.DataFrame, *, fitting: bool) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas.DataFrame.")
        if len(X) == 0:
            raise ValueError("X must contain at least one row.")
        if fitting and len(X) < 2:
            raise ValueError("X must contain at least two rows for fitting.")
        if X.shape[1] == 0:
            raise ValueError("X must contain at least one feature column.")
        if not X.columns.is_unique:
            raise ValueError("X must have unique column names.")
        return X
