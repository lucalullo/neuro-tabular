# When to choose NeuroTabular

NeuroTabular is useful when you want one neural estimator for mixed pandas data,
with local training, a scikit-learn-style API and optional internal K-fold diagnostics.
Those are library features, not claims of exclusive functionality.

The evidence below comes from a frozen six-family CPU comparison, with three
model seeds and separate default and four-candidate tuning regimes. Five methods
have complete natural-data coverage. RealMLP and TabM did not pass the declared
preliminary cold-load limit; FT-Transformer has only one fully admitted natural
result. Their missing comparisons are not predictive losses. See the
[coverage breakdown](../README.md#why-107-endpoints-are-unavailable) and
[qualification roadmap](../README.md#qualification-roadmap).

## Where the default provides a useful option

**Probability quality on Bank Marketing.** Median NLL is 0.666704 for NeuroTabular,
0.717406 for CatBoost, 0.719467 for LightGBM and 0.784755 for XGBoost. Each of the
three seed-specific paired NLL intervals favors NeuroTabular against these three
default comparators. For example, CatBoost minus NeuroTabular NLL at seed 42 is
bounded by [0.04878, 0.08058]. These are conditional, unadjusted intervals on
100-contact blocks, not evidence of superiority on other customer populations.
Median AUC is 0.719271; balanced accuracy is only 0.544196. Better NLL and ranking
do not imply an equally strong thresholded decision rule.

**Balanced decisions on the held-out Climate study.** Default median balanced
accuracy is 0.776678, compared with 0.705250 for LightGBM and 0.607143 for CatBoost.
However, CatBoost has lower median NLL (0.137218 versus 0.157654). Only one study
is held out, so these are observed results without an independent-study interval.
Choose the metric that matches your use case; this is not a general binary win.
See the [default balanced accuracy figure](results/default-binary-balanced_accuracy-training_total_cpu_s.png).

**A small final model artifact.** Default NeuroTabular model files are about
152–212 KiB on these families. On Anneal, the median is 211.78 KiB, versus
425.15 KiB for XGBoost and 3037.90 KiB for CatBoost. Small stored weights do not
mean a smaller Python/PyTorch installation or lower process RAM. XGBoost is
smaller on Climate (96.72 KiB versus 156.64 KiB); tuned NeuroTabular models can
also grow substantially.

**Large prediction batches in the two valid timing cases.** Only two of 36
dedicated A/A controls pass all frozen workload/clock requirements. For Bank
Marketing, default seed 2026, the full 9,043-row batch takes competitors
4.77× (CatBoost, CI [4.65, 4.89]), 4.69× (LightGBM, [4.65, 4.72]) and
4.19× (XGBoost, [3.96, 4.43]) the NeuroTabular time. These are instrumented
resident `predict` calls on this CPU/process pair. They are not a general
production latency claim. CatBoost is faster for one row in that same case:
its latency is 0.708× NeuroTabular, CI [0.697, 0.719].

## When another method is preferable in this panel

**Multiclass quality on Anneal and Glass.** Default Anneal balanced accuracy is
0.628742 for NeuroTabular versus 0.956947 for XGBoost, whose NLL is also much
lower (0.033408 versus 0.286735). XGBoost has lower recorded training CPU and
fit RSS. On Glass, default balanced accuracy is 0.711538 for NeuroTabular versus
0.838889 for XGBoost/LightGBM; CatBoost has the lowest median default NLL
(0.542862 versus 0.779805 for NeuroTabular). Inspect the paired intervals and
rare-class support rather than turning these two families into a universal rule.

**Regression quality when extra training cost is acceptable.** CatBoost has
median default Fiat 500 RMSE 757.44 versus NeuroTabular 786.21. On Gas NOX,
LightGBM is 12.5541 versus NeuroTabular 13.2047. LightGBM and XGBoost generally
have lower fit-compute CPU in this panel. CatBoost can cost more: its default
Anneal fit-compute median is 165.48 s versus NeuroTabular 4.69 s, with much
better classification quality. These are measured trade-offs, not equal-capacity
or equal-update comparisons.

**Lower RAM or startup requirements.** NeuroTabular does not demonstrate a
general RAM advantage here. Default Bank fit RSS is about 481 MiB versus
306 MiB for XGBoost and 338 MiB for CatBoost in the common worker. That worker
loads PyTorch even for trees, so it cannot establish standalone tree dependency,
startup or minimum-RAM requirements. See [measurement scope](MEASUREMENT_SCOPE.md).

**RealMLP, TabM or broader alternatives.** This pilot has no admitted natural-data
results for RealMLP or TabM. It cannot say when either is more accurate or more
efficient than NeuroTabular. Their CPU-limited preliminary replay outcomes do
not imply inferior predictive capability. Use separate, relevant evidence or a
separately authorized comparison; do not rank them below the measured methods.

## Default, tuning and the upgrade from 2.0.0

The four-candidate search includes four trial fits plus one final fit; equal
fit counts are not equal CPU or optimizer updates. Tuning is not always useful:
NeuroTabular Glass median NLL worsens from 0.779805 to 0.869820, and Gas NOX
RMSE from 13.2047 to 14.3009, while total training CPU rises. Anneal improves
but still trails the strongest tree results. On Bank, tuning does not preserve
the default NLL advantage over CatBoost in all seeds. `auto_tune` is not a
qualified NeuroTabular 2.1.0 feature; this is an external benchmark search.

General classifier defaults remain unchanged from 2.0.0. Their reported point
metrics match in all 24 paired classification cells. Default Fiat median RMSE
is 786.21 in 2.1.0 versus 800.49 in 2.0.0; only one default seed has a paired
interval excluding a ratio of one, and not a full 1% practical margin. Gas
predictions/metrics do not improve in this panel. This does not establish a
universal accuracy or speed gain from upgrading.

The qualified upgrade also preserves robust standardization, corrected weighted
loss reduction on CPU, optional `cv=K` diagnostics and `eval_sample_weight`.
`cv=None` keeps one standard training path; `cv=K` needs K+1 training passes,
doubled by `full_data_refit=True`, and keeps one final network. CV scores do not
automatically improve accuracy or select a recipe.

## Scope and reproducible evidence

- [Full scorecard](SCORECARD.md), [every seed and failure](results/tables/PER_SEED.csv),
  [paired intervals](results/tables/PAIRED_COMPARISONS.csv) and
  [all local Pareto frontiers](results/decision_support.json).
- [Default binary paired NLL](results/intervals/default-binary-primary-paired-intervals.png),
  [multiclass](results/intervals/default-multiclass-primary-paired-intervals.png) and
  [regression](results/intervals/default-regression-primary-paired-intervals.png).
- [Source identities, statistical units and licenses](DATASETS.md),
  [methods](METHODS.md), [protocol](PROTOCOL.md) and [reproduction](README.md).

The 34 inconclusive A/A controls prohibit broad warm-latency conclusions.
The [timing audit](MEASUREMENT_SCOPE.md#inference-timing-control-audit) documents
the clock/interval limitations and the required follow-up qualification.
Quality uncertainty is conditional on frozen models and declared TEST units;
it is not independent-family uncertainty and is not multiplicity-adjusted.
The panel has two families per task, not 18 independent datasets. It is not an
independent confirmation of a new classification recipe. The historical
LightGBM/GrassBase anomaly is not used in this comparison.

The delivered default supports at most 128 categorical columns through its
128-dimension aggregate embedding budget; this is not a limit on numeric or
total columns and not a byte-level RAM guarantee. Extended categorical support,
warm-start, automatic tuning and experimental calibration remain excluded.
CUDA on physical NVIDIA hardware and formal end-to-end no-network certification
remain unqualified. Local training without mandatory pretrained weights is a
library property, not an exclusivity or completed security-certification claim.
