# Sources, statistical roles and limitations

The six sources were selected before model results for task and feature
diversity. Known sources used for predictive tuning in the reviewed history
were excluded. Available legacy history is incomplete, so absence of a match
is not a worldwide “never used” guarantee. This public frozen-release pilot
is not the independent confirmation of a newly selected algorithm.

[prepared_sources.json](prepared_sources.json) records exact source hashes,
feature names, missing-value counts, class distributions, split sizes and role
hashes for each seed. The raw files are not redistributed here. The loader
checks source identity and hashes before parsing; no model output is involved
in source preparation or split choice.

| Family | Task | Original rows | Predictors | Classes | Source and license |
|---|---|---:|---:|---:|---|
| Bank Marketing, `bank-full` | Binary | 45,211 | 13 | 2 | [UCI 222](https://doi.org/10.24432/C5K306), CC BY 4.0 |
| Climate Model Simulation Crashes | Binary | 540 | 18 | 2 | [UCI 252](https://doi.org/10.24432/C5HG71), CC BY 4.0 |
| Anneal, curated v18 | Multiclass | 898 | 38 | 5 present | [UCI 3](https://doi.org/10.24432/C5RW2F), [OpenML 46906 v18](https://www.openml.org/d/46906), CC BY 4.0 |
| Glass Identification | Multiclass | 214 | 9 | 6 present | [UCI 42](https://doi.org/10.24432/C5WW2P), CC BY 4.0 |
| Gas Turbine NOX, 2011–2015 | Regression | 36,733 | 9 | — | [UCI 551](https://doi.org/10.24432/C5WC95), CC BY 4.0 |
| Used Fiat 500 | Regression | 1,538 | 7 | — | [Original Kaggle source](https://www.kaggle.com/datasets/paolocons/another-fiat-500-dataset-1538-rows), [OpenML 46907 v1](https://www.openml.org/d/46907), CC0 / public domain |

Credit belongs to the original contributors listed by those sources. The
source licenses remain separate from the MIT license of this repository.
The curated variants, exclusions, grouped roles and hashes below are part of
this benchmark; their scores must not be compared directly with unrelated
leaderboards using other features or splits.

## Binary families

**Bank Marketing:** the original chronologically ordered `bank-full.csv`.
Target `y`. Call duration is excluded because it is unavailable before the
call outcome. Day and month are also predeclared exclusions in this 13-feature
variant. The first 60% of ordered contacts form FIT, the next 20% validation,
and the last 20% TEST. Purging earlier exact input duplicates removes 111
records: final FIT 27,024, validation 9,033 and TEST 9,043. The roles are fixed
across model seeds. Client identifiers are unavailable, and repeated contacts
can concern the same person. This measures later-contact performance, not
independence across new clients. Uncertainty uses 100 consecutive contacts;
longer unobserved client dependencies remain a limitation.

**Climate:** simulation output, not observational patient/person data.
Three 180-run Latin-hypercube studies vary 18 POP2 parameters. Study 1 is FIT,
study 2 validation, study 3 TEST; study/run IDs are not features. Outcome
retains the source orientation (failure 0, success 1). Repeated model seeds
reuse these roles. There is just one held-out study, so row bootstrap is not
presented as uncertainty across independent simulation studies.

## Multiclass families

**Anneal:** the pinned OpenML/TabArena variant merges the original source
files, represents source “not applicable” values explicitly, and contains five
classes; absent source class 4 is not invented. OpenML task 363614 supplies
the first three official repeat/fold/sample tuples. All three cross exact
input-signature groups, so the predeclared stratified-group fallback applies.
FIT/validation/TEST sizes are 537/180/181, 541/179/178 and 541/180/177 for
seeds 42/2026/1947. Unreported industrial batch relationships remain unknown.

**Glass:** ID is excluded, target is `Type_of_glass`. Exact input signatures
are grouped in five stratified folds: fold 0 TEST, fold 1 validation, the other
three FIT. Sizes are 128/43/43, 128/44/42 and 128/43/43. Glass deliberately
adds a tiny-data case below the earlier unexecuted proposal's 500-row minimum;
this scope change was made before any model result. Small rare classes and
unreported production-batch IDs limit precision and independence claims.

## Regression families

**Gas NOX:** target `NOX`; year and the co-target `CO` are excluded. The nine
sensor/generation features describe co-measured emissions estimation, not a
new-plant forecast. FIT uses 2011–2012 (15,039 records), validation 2013
(7,152), and TEST 2014–2015 (14,542). Thus final training uses the first three
years, consistent with the source recommendation. Unit bootstrap uses blocks
of 168 consecutive observed records within each year; these are not asserted
to be 168 consecutive calendar hours. This is one Turkish power plant, related
to UCI 294, so the two sources could not count as independent families.

**Fiat 500:** target `price`. The pinned public listing variant has no vehicle
VIN, dealer identifier or reliable acquisition-time grouping. OpenML task
363615's first three official folds overlap exact input signatures; the
predeclared grouped fallback therefore applies. FIT/validation/TEST sizes are
923/307/308, 919/308/311 and 924/306/308. Exact signatures prevent obvious
duplicate leakage, but cannot prove independence between vehicles/dealers or
remedy missing source lineage. Do not claim new-dealer generalization.

## Shared safeguards

All splits retain their original target orientation. No probability inversion,
test-label encoding choice, threshold tuning or label-dependent row removal is
allowed. Grouping uses exact typed input equality, joined with declared groups
where present. Both target variants and descendants of a source share lineage.
Preprocessing is fitted only on the current training role. All roles must have
the recorded class support before fits; a violation stops rather than redraws.

Accuracy, AUC, probability quality, speed and memory answer different questions.
Neither favorable NLL nor these six sources establish that a library is best
for all tabular data. The final report will retain adverse results and errors.
