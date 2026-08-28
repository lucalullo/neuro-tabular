"""Leakage-safe preprocessing for heterogeneous pandas DataFrames."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from hashlib import blake2b
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

    ``max_categories`` and ``hash_buckets`` are research-only controls used
    by the release ablations. They are intentionally not exposed by the
    estimator's public constructor.
    """

    def __init__(
        self,
        categorical_features: Iterable[Hashable] | None = None,
        min_category_count: int = 2,
        *,
        numeric_strategy: str = "standard",
        smooth_clip_limit: float = 5.0,
        n_numeric_bins: int = 8,
        use_category_frequency: bool = False,
        max_categories: int | None = None,
        hash_buckets: int = 0,
    ) -> None:
        self.categorical_features = categorical_features
        self.min_category_count = min_category_count
        self.numeric_strategy = numeric_strategy
        self.smooth_clip_limit = smooth_clip_limit
        self.n_numeric_bins = n_numeric_bins
        self.use_category_frequency = use_category_frequency
        self.max_categories = max_categories
        self.hash_buckets = hash_buckets

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
        self.rare_category_ids_: dict[Hashable, dict[object, int]] = {}
        self.category_id_frequencies_: dict[Hashable, np.ndarray] = {}
        self.categorical_cardinalities_: list[int] = []
        for column in self.categorical_features_:
            series = X[column]
            try:
                counts = series.value_counts(dropna=True, sort=False)
                eligible = [
                    value
                    for value, count in counts.items()
                    if int(count) >= self.min_category_count
                ]
                rare = {
                    value
                    for value, count in counts.items()
                    if int(count) < self.min_category_count
                }
                if (
                    self.max_categories is not None
                    and len(eligible) > self.max_categories
                ):
                    ranked = sorted(
                        eligible,
                        key=lambda value: (-int(counts[value]), repr(value)),
                    )
                    retained = set(ranked[: self.max_categories])
                    rare.update(value for value in eligible if value not in retained)
                    eligible = [value for value in eligible if value in retained]
                frequent_offset = 3 if self.hash_buckets == 0 else 2 + self.hash_buckets
                vocabulary = {
                    value: index + frequent_offset
                    for index, value in enumerate(eligible)
                }
            except TypeError as exc:
                raise ValueError(
                    f"Categorical column {column!r} contains unhashable values."
                ) from exc
            self.category_vocabs_[column] = vocabulary
            self.rare_categories_[column] = rare
            self.rare_category_ids_[column] = {
                value: RARE_CATEGORY_ID + self._stable_bucket(value, self.hash_buckets)
                for value in rare
            }
            frequency_scale = np.log1p(max(1, len(series)))
            cardinality = len(vocabulary) + frequent_offset
            id_frequencies = np.zeros(cardinality, dtype=np.float32)
            id_frequencies[MISSING_CATEGORY_ID] = float(
                np.log1p(int(series.isna().sum())) / frequency_scale
            )
            if self.hash_buckets:
                bucket_counts = np.zeros(self.hash_buckets, dtype=np.int64)
                for value in rare:
                    bucket = self.rare_category_ids_[column][value]
                    bucket_counts[bucket - RARE_CATEGORY_ID] += int(counts[value])
                id_frequencies[
                    RARE_CATEGORY_ID : RARE_CATEGORY_ID + self.hash_buckets
                ] = np.log1p(bucket_counts) / frequency_scale
            else:
                rare_count = sum(int(counts[value]) for value in rare)
                id_frequencies[RARE_CATEGORY_ID] = float(
                    np.log1p(rare_count) / frequency_scale
                )
            for value, category_id in vocabulary.items():
                id_frequencies[category_id] = float(
                    np.log1p(int(counts[value])) / frequency_scale
                )
            self.category_id_frequencies_[column] = id_frequencies
            self.categorical_cardinalities_.append(cardinality)
        categorical_finished = perf_counter()

        self.n_continuous_features_ = len(self.numeric_features_)
        self.n_frequency_features_ = (
            len(self.categorical_features_) if self.use_category_frequency else 0
        )
        self.n_numeric_outputs_ = (
            2 * self.n_continuous_features_ + self.n_frequency_features_
        )
        self.fit_sample_count_ = len(X)
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
        categorical, category_frequency = self._transform_categorical(X)
        categorical_finished = perf_counter()
        if self.use_category_frequency:
            numerical = np.concatenate((numerical, category_frequency), axis=1)
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
            self.numeric_knots_ = np.empty(
                (0, self.n_numeric_bins + 1), dtype=np.float32
            )
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
        scaled = (imputed - centers) / scales
        if self.numeric_strategy == "robust":
            limit = self.smooth_clip_limit
            scaled = limit * np.tanh(scaled / limit)
        quantiles = np.linspace(0.0, 1.0, self.n_numeric_bins + 1)
        knots = np.quantile(scaled, quantiles, axis=0).T
        self.numeric_knots_ = self._stabilize_knots(knots)

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

    def _transform_categorical(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n_samples = len(X)
        if not self.categorical_features_:
            return (
                np.empty((n_samples, 0), dtype=np.int64),
                np.empty((n_samples, 0), dtype=np.float32),
            )
        categorical = np.empty(
            (n_samples, len(self.categorical_features_)), dtype=np.int64
        )
        frequencies = (
            np.empty_like(categorical, dtype=np.float32)
            if self.use_category_frequency
            else np.empty((n_samples, 0), dtype=np.float32)
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
                    if self.hash_buckets:
                        mapped_rare = series.map(self.rare_category_ids_[column])
                        rare_ids = mapped_rare.to_numpy(
                            dtype=np.float64, na_value=np.nan
                        )
                        rare_mask = np.isfinite(rare_ids)
                        encoded[rare_mask] = rare_ids[rare_mask]
                    else:
                        encoded[series.isin(rare).to_numpy(dtype=bool)] = (
                            RARE_CATEGORY_ID
                        )
            except TypeError as exc:
                raise ValueError(
                    f"Categorical column {column!r} contains unhashable values."
                ) from exc
            encoded[missing] = MISSING_CATEGORY_ID
            encoded_ids = encoded.astype(np.int64, copy=False)
            categorical[:, index] = encoded_ids
            if self.use_category_frequency:
                frequencies[:, index] = self.category_id_frequencies_[column][
                    encoded_ids
                ]
        return categorical, frequencies

    @staticmethod
    def _stabilize_knots(knots: np.ndarray) -> np.ndarray:
        """Return finite, strictly increasing per-feature quantile knots."""

        stable = np.asarray(knots, dtype=np.float64).copy()
        for feature in range(stable.shape[0]):
            row = stable[feature]
            if not np.isfinite(row).all():
                raise ValueError("Numerical quantiles produced non-finite knots.")
            scale = max(1.0, float(np.max(np.abs(row))))
            epsilon = 1e-6 * scale
            if row[-1] - row[0] < epsilon:
                center = float(row[0])
                row[:] = np.linspace(center - 0.5, center + 0.5, len(row))
                continue
            for index in range(1, len(row)):
                row[index] = max(row[index], row[index - 1] + epsilon)
        return stable.astype(np.float32)

    @staticmethod
    def _stable_bucket(value: object, bucket_count: int) -> int:
        if bucket_count < 1:
            return 0
        payload = f"{type(value).__qualname__}:{value!r}".encode(
            "utf-8", errors="backslashreplace"
        )
        digest = blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little") % bucket_count

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
        if (
            not isinstance(self.n_numeric_bins, Integral)
            or isinstance(self.n_numeric_bins, bool)
            or self.n_numeric_bins < 2
        ):
            raise ValueError("n_numeric_bins must be an integer of at least 2.")
        if not isinstance(self.use_category_frequency, bool):
            raise TypeError("use_category_frequency must be a boolean.")
        if self.max_categories is not None and (
            not isinstance(self.max_categories, Integral)
            or isinstance(self.max_categories, bool)
            or self.max_categories < 1
        ):
            raise ValueError("max_categories must be None or a positive integer.")
        if (
            not isinstance(self.hash_buckets, Integral)
            or isinstance(self.hash_buckets, bool)
            or self.hash_buckets < 0
        ):
            raise ValueError("hash_buckets must be a non-negative integer.")

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
