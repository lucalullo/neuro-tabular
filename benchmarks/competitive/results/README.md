# Measured benchmark results

This directory contains the frozen, measured six-family CPU pilot, not a
synthetic example. [status.json](status.json) records 181 measured final TEST
endpoints out of 288 planned (107 unavailable). Five methods have complete
coverage; RealMLP and TabM have none and FT-Transformer has one. The unavailable
endpoints are 96 backend qualification exclusions, six cold replay stops and
five measurement-pair stops. See the [coverage table](../../README.md#why-107-endpoints-are-unavailable).

NeuroTabular 2.0.0 is a version reference, not an external comparator in the
summary overview. Default and four-candidate tuning remain separate.

- [Full results JSON](results.json) and [CSV](results.csv): metrics, uncertainty,
  costs, status, hashes and retained failures.
- [Summary](summary.json): descriptive family-weighted summaries and coverage.
- [Frozen report](REPORT.md): original measured tables and interpretation.
- [All per-seed results](tables/PER_SEED.csv), [family summaries](tables/FAMILY_SUMMARY.csv),
  [paired comparisons](tables/PAIRED_COMPARISONS.csv) and [timing controls](tables/TIMING_CONTROLS.csv).
- [Detailed figures](../../FIGURES.md) and [selection guide](../WHEN_TO_CHOOSE.md).
- [Decision support](decision_support.json): local descriptive Pareto frontiers.

Only two of 36 inference A/A controls passed; see the [timing audit](../MEASUREMENT_SCOPE.md#inference-timing-control-audit).
These results support dataset- and metric-specific observations, not universal
superiority or a general speed claim. Incomplete methods are not ranked on
surviving subsets.

The schema is [schema.json](schema.json). See the [reproduction guide](../README.md),
[protocol](../PROTOCOL.md) and [custody record](../POSTPROCESSOR_CUSTODY.json).
The original reports and data are retained; editorial coverage and timing
analysis do not introduce new fits, inference, scoring or measurements.
