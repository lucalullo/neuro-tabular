# Methods and resource profile

All versions and official-wheel SHA256 values are in [software.json](software.json)
and the optional Windows CPython 3.12 lock. No competitor is a runtime
dependency of NeuroTabular. Every method has one final persisted predictor.

| Method | Implementation | Frozen version | Controls |
|---|---|---|---|
| NeuroTabular 2.0.0 | Original wheel | 2.0.0 | `random_state=seed` only |
| NeuroTabular 2.1.0 | Frozen qualified runtime | 2.1.0 | `random_state=seed` only |
| RealMLP | PyTabKit `RealMLP_TD_Classifier/Regressor` | PyTabKit 1.8.0 | CPU, 1 thread, 1 CV model, 0 refits, 1 repeat, no extra calibration |
| XGBoost | `XGBClassifier/Regressor` | 3.4.1 | 1 thread, seed, native categoricals enabled |
| LightGBM | `LGBMClassifier/Regressor` | 4.7.0 | 1 thread, seed, native categoricals, quiet log |
| CatBoost | `CatBoostClassifier/Regressor` | 1.2.10 | CPU, 1 thread, seed, no auxiliary output files |
| TabM | PyTabKit `TabM_D_Classifier/Regressor`, bundled implementation | PyTabKit 1.8.0 | Same PyTabKit resource controls; joint k=32 preset is disclosed |
| FT-Transformer | PyTabKit `FTT_D_Classifier/Regressor` | PyTabKit 1.8.0 | Same PyTabKit resource controls |

“Default” means the named preset under these explicit common CPU controls,
not each library's unconstrained all-core or multi-refit regime. TabM's joint
members are part of that competitor's preset and their complete cost is
charged. NeuroTabular remains one original neural network, without HPO or
ensembling inside its estimator. No pretrained weights are used.

## Feature adapters

NeuroTabular receives the pandas inputs directly and uses only its native
preprocessing. CatBoost uses typed string tokens for missing/categories.
XGBoost and LightGBM use pandas categorical columns with TRAIN-only vocabulary;
unseen values become native missing categories. XGBoost alone requires an
explicit target-code mapping, reversed for prediction and probability columns.
PyTabKit gets TRAIN-only numeric median imputation where necessary and its
native categorical handling. These adapters and their fit/inference costs are
included in measurement. There is no TF-IDF, target encoding, hand-selected
feature enhancement or shared preprocessing trained on TEST.

## Four-candidate search

The exact list, order and values for every method are in `hpo_candidates` in
[protocol.json](protocol.json). Candidate zero is the resource-controlled
default. The other three are fixed before results; no adaptive search,
dataset-specific space or outcome-driven replacement is allowed. Every trial
gets the same 225 s CPU cap; the final fit gets 300 s. Equal numbers of fits
are not asserted to be equal actual compute. Both realized cost and failures
are reported.

The local constructor/clone audit covers all 96 method/task/candidate settings
without fitting. It does not substitute for the proposed synthetic-data fit
and persistence gate or for measured results on public data.

Primary implementation references:
[RealMLP / PyTabKit](https://github.com/dholzmueller/pytabkit),
[XGBoost](https://xgboost.readthedocs.io/en/stable/),
[LightGBM](https://lightgbm.readthedocs.io/en/stable/),
[CatBoost](https://catboost.ai/docs/),
[TabM](https://github.com/yandex-research/tabm),
[FT-Transformer](https://github.com/yandex-research/rtdl-revisiting-models).
The optional environment's packages retain their respective licenses; they
are installed from official wheels, not copied into this repository.
