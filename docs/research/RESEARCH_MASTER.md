# Current Product State

**PRODUCT TRUTH — NeuroTabular 0.3.0 is the current release target and a locally frozen release candidate.** The previous public release is v0.2.0. The candidate supports binary classification, multiclass classification and regression through a shared neural core. It is a compact pandas/scikit-learn-style neural estimator for mixed tabular data, with automatic preprocessing and validation-based checkpoint selection.

The current branch is `release/0.3.0`, commit `b3bda31ac86fd4a0bb011046615cc5bac79e1033`. Product development closed at `dev/0.3.0`, `7d2e00d4ee52c6289db3a1598b8589a87b03bd2a`; the release branch derives from that product line, not from H23. The release audit verdict is **0_3_RELEASE_CANDIDATE_FROZEN**, and the subsequent local delivery receipt is **LOCAL_0_3_DELIVERY_READY**. Neither means publication occurred. Remote CI is **NOT EXECUTED** in the available delivery evidence, physical CUDA validation is **PENDING**, and historical held-outs remain **CLOSED**.

| Product property | Frozen behavior |
| --- | --- |
| Backbone | Pre-normalized residual MLP, hidden width 64, two residual blocks, SiLU, dropout 0.1 |
| Training defaults | AdamW, learning rate 0.003, weight decay 1e-5, cosine schedule, automatic batching, maximum 30 epochs |
| Selection | Validation loss, patience **4**, min_delta 1e-4, evaluation every epoch; full_data_refit=False |
| Numeric preprocessing | Training-only median imputation, standard scaling and missing indicators; scalar representation by default |
| Categorical preprocessing | Automatic object/string/bool/category discovery, with explicit categorical columns available; separate missing/unknown/rare IDs, rare threshold 2, learned embeddings and training-only log-frequency side features |
| Classifier | `NeuroTabularClassifier`: binary one-logit BCE/sigmoid; multiclass cross-entropy/softmax and ordered original class labels |
| Regressor | `NeuroTabularRegressor`: MSE with training-only weighted target moments; predictions restored to original units; constant targets preserved exactly |
| Input/API boundary | pandas DataFrame with unique column names, single output; column reordering accepted, missing/extra columns rejected; sklearn cloning, parameters, pipeline/CV and task-specific score |
| Weights | Zero-weight rows removed before discovery/split/preprocessing; safe positive-weight normalization; class balancing is classification-only |
| Persistence | Local fresh-process pickle/joblib and installed wheel/sdist checks passed; cross-version/device compatibility is not guaranteed |
| Validation | Archived release checks: **147 passed, 3 physical-CUDA skips, 0 failures**; Ruff and clean wheel/sdist installation passed |
| Limits | No calibrated-probability API, same-process concurrent-fit guarantee, physical GPU performance claim or universal advantage over trees |

**RESEARCH ARCHIVE** records experiments and conditional signals, not current public defaults. GELU, AVG3, INPUT_GATE, BOOST, B_STRONG, MIMO and tree-guided architectures were **not promoted into 0.3.0**. Inherited optional numerical embeddings and `feature_gating` already existed in the public v0.2 line; they must not be confused with promotion of H12's contextual INPUT_GATE. There is no public activation-selection constructor argument.

Old duplicate version labels are **LEGACY / HISTORICAL INTERNAL LINEAGE** and do not supersede the current public v0.2.0 → target0.3.0 history.

This memory was reconstructed on 2026-09-20 from local Git objects and retained documents. It introduces no fit, metric recomputation from raw data, held-out access, remote operation, product change or new research authorization. Read README and these two memory files first. Retrieve full archives only to answer a specific evidentiary question.

# Scientific Rules

1. Real varied observational datasets outrank synthetic diagnostics; label simulator-origin kin8nm separately.
2. Freeze hypothesis, eligible panels, seed roles, controls, metrics, stopping and gates before outcomes.
3. Pair train/validation/test rows and seed roles; fit preprocessing, vocabularies and target statistics on training only.
4. Use fresh confirmation seeds; distinguish fresh seeds from fresh datasets and historical replay.
5. No post-hoc gate changes, dataset exclusions, extra seeds, objective switches or one-dataset tuning to rescue failures.
6. Preserve negative, invalid, contaminated and failed attempts with reasons; unexecuted is not negative.
7. Algorithm claims require compute-matched controls from the first screen; parameter matching is separate.
8. Charge teacher, selector, pretraining and retrieval state; separate training cost, inference cost and memory.
9. Report task-native metrics, medians, breadth, worst dataset/seed, probability quality and contributor sensitivity.
10. Average seeds within dataset then weight datasets explicitly; avoid pooling correlated datasets and repeated evidence.
11. Preserve exact raw/source/data/split/checkpoint provenance; replay predictions and recompute metrics for scientific audits.
12. Separate research candidates from public API/default changes; no unsupported SOTA, novelty or universal claims.
13. Keep all six reserved held-outs closed during development; future opening requires frozen protocol and separate authorization.

**Metric discipline.** Δ means candidate minus named comparator unless a favorable error reduction is explicitly stated. Higher AUC/accuracy/F1/R² is better; lower logloss/Brier/RMSE/MAE is better. Average seeds within datasets, then datasets with declared weights. Ranking is not calibration. A positive mean cannot override a failed median, worst-case, integrity or cost gate.

H12 NG = primary delta / max(0.05, 1−B0 primary), with AUC/accuracy/R² as primary. H13–H23 NG uses 2×ΔAUC, Δaccuracy/(1−1/K), or RMSE improvement divided by training-target SD. **These scales are not interchangeable.** Product relative RMSE instead divides by the test RMSE of the training-mean predictor. B0/B1/C1/BC/TC are phase-local names: H13 B1 is INPUT_GATE, H14 B1 is BOOST, H18 C1 is H15 C4. A cost-allocated baseline may use a different horizon and cosine schedule; it is not necessarily a better score on every dataset.

# Held-out Datasets

**CLOSED.** The reserved historical names are:

| Dataset | OpenML ID | Reserved historical seeds |
| --- | --- | --- |
| heart-statlog | UNKNOWN / NOT RECONSTRUCTED | UNKNOWN / NOT RECONSTRUCTED |
| credit-approval | UNKNOWN / NOT RECONSTRUCTED | UNKNOWN / NOT RECONSTRUCTED |
| cylinder-bands | UNKNOWN / NOT RECONSTRUCTED | UNKNOWN / NOT RECONSTRUCTED |
| kr-vs-kp | UNKNOWN / NOT RECONSTRUCTED | UNKNOWN / NOT RECONSTRUCTED |
| kc1 | UNKNOWN / NOT RECONSTRUCTED | UNKNOWN / NOT RECONSTRUCTED |
| mammography | UNKNOWN / NOT RECONSTRUCTED | UNKNOWN / NOT RECONSTRUCTED |

Never download, inspect outcomes, evaluate, tune or select development hypotheses using these datasets or aliases. The archived Micro-Lab protocol explicitly treats Australian as a forbidden credit-approval alias. Opening requires a frozen candidate and protocol plus separate authorization. Once evaluated, do not retouch the model using held-out outcomes.

The preserved documents verify the six names but do not reconstruct their original ID/seed reservation manifest. Micro-Lab excludes 61/73/89; that sentence alone does not establish the current reserved mapping, and 73 was later used in H5/H6/H6C development. JSON uses null rather than an invented reservation. The seven held-outs reported in legacy Cycle2 were a **different set**, including datasets later authorized as development; their historical evaluation does not imply these six were opened.

# Phase History H1–H23

Verdicts are historical decisions, not present product approval. H1 has no recovered standalone formal verdict token; its documented suite-freeze result is preserved without inventing one. Commits below are verified local closure branch tips; the full SHA/reference table follows. H1–H4 share the first-campaign closure.

| Phase | Hypothesis | Main experiment | Key result | Verdict | Knowledge gained |
| --- | --- | --- | --- | --- | --- |
| H1 (`46b165f2`) | Freeze real development evidence | Frozen development manifest | 13 binary datasets; 3 seeds; 39 paired HGB context fits | UNKNOWN / NOT RECONSTRUCTED | Training-only data/splits established; no standalone formal H1 verdict token found. |
| H2 (`46b165f2`) | GELU activation-only substitution | Frozen paired development protocol | 39 pairs; mean AUC +0.003103; median +0.001096; 11 W/1 L/1 T | PROMISING_FOR_COMPOSITION | Initial signal only; later fresh-seed default claim failed. |
| H3 (`46b165f2`) | Microbatch/update-frequency vs accumulation and LR control | Frozen paired development protocol | 13 datasets; selected-checkpoint AUC +0.003396; without Phoneme/Bank -0.00003027 | SPECIALIZED_SIGNAL | Update dynamics interact with LR; no general microbatch policy. |
| H4 (`46b165f2`) | AUC checkpoint plus robust scalar calibration | Frozen paired development protocol | 15 trajectories; calibrated AUC delta +0.000153; logloss +0.000688 | REJECT | No general ranking/probability improvement; four-way splits matter. |
| H5 (`937c3b8e`) | Activation confirmation and seven component screens | Frozen paired development protocol | 65 GELU pairs; mean AUC +0.004540; median +0.001596; 10 W/3 L | GELU_CONFIRMED_FOR_COMPOSITION | Only GELU retained; GEGLU/interaction/normalization/width/depth forms rejected. |
| H6 (`89de66de`) | Broaden GELU promotion panel | Frozen paired development protocol | 22-real post-hoc view; 102 pairs; mean AUC +0.004314; 18 W/4 L | GELU_REMAINS_RESEARCH_ONLY | Numerical gates passed but Madelon synthetic inclusion violated real-data protocol. |
| H7 (`d2173ef7`) | Init/split diagnosis; patience 4 to 6 | Frozen paired development protocol | Primary 16 datasets; AUC +0.000184639; median 0; 13/14 gates pass | NO_GENERAL_STABILITY_MECHANISM_YET | Positive-median gate failed; blood worsened and credit-g variance rose. |
| H8 (`f8f6de84`) | Separate init, train membership and validation membership | Frozen paired development protocol | 91 unique fits; 4 MIXED / 2 STABLE / 1 INIT_DOMINATED; 0 intervention fits | MIXED_SPLIT_SENSITIVITY | No common factor justified a stability intervention. |
| H9 (`4dfad11c`) | Average top-three validation-loss checkpoints | Frozen paired development protocol | 15-dataset generalization; AUC +0.003182; median +0.000221; variance ratio 1.484613 | SPECIALIZED_CHECKPOINT_AVERAGING | Quality signal failed variance gate, heavily influenced by Amazon. |
| H10 (`cf36ed51`) | One-block feature-token attention with capacity control | Frozen paired development protocol | 48 fits; T2 AUC +0.016397; median +0.008990; diabetes -0.011019 | REPRESENTATION_NOT_SUPPORTED | Worst-dataset bound failed; official timing unavailable; no attention ablation. |
| H11 (`7767768e`) | Offline scope of H10 numeric/categorical effects | Offline evidence analysis | 8 datasets; 48 H10 raw fits reanalyzed; 0 new fits | NO_HYBRID_HYPOTHESIS_JUSTIFIED | No systematic numerical harm localized a justified hybrid; architecture untested. |
| H12 (`b44ec304`) | 15 canonical multi-task challengers | Frozen paired development protocol | INPUT_GATE 39 datasets; NG +0.010461; median +0.003130; 29 W/5 L/5 T | NEW_BACKBONE_CANDIDATE_FOUND | Frozen gates passed, modest concentrated development signal; no product integration. |
| H13 (`6026fc17`) | 22 operators plus AutoPrep references | Frozen paired development protocol | BOOST 39 datasets; NG +0.011627; median +0.005144; 31 W/8 L | PROMISING_ALGORITHM_FOUND | Worst-case/capacity limits blocked strong claim; AutoPrep zero median. |
| H14 (`5b95bc25`) | Fresh BOOST, capacity and compute controls | Frozen paired development protocol | 39 datasets; BOOST-B0 NG +0.010739; BOOST-LONG -0.007930; 14 W/22 L/3 T vs LONG | BOOST_NOT_CONFIRMED | Quality replicated; extra compute explains or exceeds advantage; mechanism gate failed. |
| H15 (`920c2ffc`) | 20-40 epoch frontier; global 35-epoch confirmation | Frozen paired development protocol | 39 datasets; NG +0.011244; regression R2 +0.018259; isolated fit 1.657719× | COMPUTE_SCALING_SIGNAL | Only regression practical gate passed; B_STRONG not general default. |
| H16 (`8d12dec6`) | 17 second-generation forms vs immediate compute controls | Frozen paired development protocol | 457 fits; MIMO mean NG +0.015196; median -0.010379; 4 W/8 L | NO_NEW_MECHANISM | MIMO catastrophe/concentration; no candidate reached Stage B. |
| H17 (`fbf9afbc`) | Offline MIMO safe-routing plausibility | Offline evidence analysis | 12 datasets; 2 catastrophic losses; 0 new fits; no qualifying universal signal | NO_SAFE_SPECIALIST_SIGNAL | NO_SAFE_ROUTING_HYPOTHESIS; SAFE not built; MIMO closed. |
| H18 (`5ce2fac0`) | C0/C1 quality oracle before meta-policy | Offline evidence analysis | 39 datasets; oracle-over-global NG +0.001350; median 0; 7/39 above 0.002 | NO_META_ADAPTATION_HEADROOM | Quality headroom gate failed for this pool; no policy trained. |
| H19 (`cd5945db`) | PC, PFN prototype, relaxed logic, DEQ | Frozen paired development protocol | 351 recorded fits; 24 original PC invalid; DEQ B NG -0.055765; median -0.028443 | NO_RADICAL_FAMILY_SIGNAL | Corrected PC/logic/PFN fail; DEQ binary hint failed B confirmation. |
| H20 (`410dba22`) | Fixed CatBoost OOF targets; alpha 0.5 neural loss | Frozen paired development protocol | 12 datasets; NG vs BC +0.002121; median -0.001376; 5/12 nonnegative | NO_TREE_DISTILLATION_SIGNAL | No median transfer; confidence and leaf auxiliary variants not triggered. |
| H21 (`5aa4c4d3`) | Tree soft-threshold basis and sparse pair products | Frozen paired development protocol | 12 datasets; NG vs BC +0.058229; T1-T2 -0.000267; T1-T3 +0.008427 | TREE_INTERACTION_SIGNAL | DESCRIPTIVE_STAGE_A_ONLY; primary gate FAIL; candidate NOT_PROMOTED. |
| H22 (`0c362905`) | Prospective graph-only basis with matched controls | Frozen paired development protocol | 18 datasets; NG vs GC -0.020484; median -0.007202; 6/18 nonnegative | NO_TREE_GRAPH_SIGNAL | Prospective and overlap fail; random-graph contrast depends on ionosphere. |
| H23 (`60ae0ddd`) | Eight training-only forms; fresh regression policy confirmation | Frozen paired development protocol | 341 scientific fits; 0 survivors; regression R2 +0.019427; fit CPU 2.295031× | RELEASE_WITH_CURRENT_BACKBONE | Training screens fail vs TC; 35-epoch policy exceeds 1.8× release cost bound. |

H6C and CORE GELU are supplemental milestones between H6 and H7, not missing numbered phases:

| Milestone | Exact verdict | Key result |
| --- | --- | --- |
| H6C (`8302265f`) | GELU_DEFAULT_CANDIDATE_CONFIRMED | 22 real datasets, 102 pairs, 204 fits; mean AUC +0.004314079, median +0.001753404, 18 W/4 L; identical H6 eligible raw metrics, not independent replication. |
| CORE_GELU (`0245f14d`) | GELU_CORE_CANDIDATE_NOT_READY | Fresh seeds211/307/401:22 datasets, 132 summary fits; mean AUC -0.000070114801, median -0.000046670436, 10 W/12 L, worst credit-g -0.045873015873. FAILED_INTEGRITY: original raw/source/split evidence missing. Separate36-fit equivalence recovery passed; does not repair original confirmation. |

# Important Phase Details

## GELU: preserve the complete decision chain

H2 changed only the activation, with paired initial parameters/RNG, and found mean ΔAUC +0.003103 on 39 pairs. H5 extended the 13-dataset panel to five seeds: +0.004540 mean, +0.001596 median, 10 dataset wins and three losses; removing the best two left +0.002049. Its seven additional component forms all failed their general gates. There was no independently supported second component to stack.

H6 was numerically positive but protocol-invalid for promotion: artificial Madelon entered the supposed real-data campaign and was recognized after six fits. The eligible 22-real-dataset view remained a post-hoc subset, with 102 pairs and +0.004314 mean AUC. Retaining invalid/excluded evidence was correct; rewriting the eligibility rule would not have been. **GELU_REMAINS_RESEARCH_ONLY** was a compliance decision, not a claim of negative numerical performance.

H6C preregistered the clean 22-real panel and reran 204 fits. Mean ΔAUC +0.004314079, median +0.001753404, 18 W/4 L, worst diabetes −0.000740741; median fit ratio 1.046538. It earned **GELU_DEFAULT_CANDIDATE_CONFIRMED** under those gates. However, eligible H6 and H6C metric arrays reproduce exactly with the same seeds/splits. They are **not two independent replications**.

CORE GELU then used fresh seeds 211/307/401 across 22 datasets: 132 reported fits, mean ΔAUC −0.000070114801, median −0.000046670436, 10 W/12 L, worst credit-g −0.045873015873. Five numerical subcriteria failed. The original raw predictions, source bindings, exact split evidence and runner were not retained, so the original confirmation also carries **FAILED_INTEGRITY**. These numbers are preserved as report-level evidence, not independently certified raw results.

A separate 36-fit recovery established SiLU regression freeze and research/core GELU equivalence on its restricted panel. It does not reconstruct missing original evidence or establish pre-result preregistration. The original field called “updates” was rows × epochs, not optimizer steps. Final decision: **GELU_CORE_CANDIDATE_NOT_READY**; present scientific classification **NOT_SUPPORTED_AS_GENERAL_DEFAULT**. Keep SiLU.

## H7–H8: stability is heterogeneous

H7's factorial diagnosis crossed initialization and split on six datasets while fixing the test and training RNG. Credit-g was split-dominated in that design; Titanic, diabetes and ionosphere were mixed, while electricity was stable in absolute variation. The selected single intervention was patience 4→6 at the same 30-epoch horizon. The primary 16-dataset confirmation gave mean ΔAUC +0.000184639 but exactly zero median. It passed 13/14 gates and failed the required positive median. Blood-transfusion lost −0.009502924 AUC; credit-g's derivation-panel variance increased 7.55×. Forty-eight of 66 paired returned predictions were unchanged. Reduced aggregate variance partly reflected worse quality, not a general remedy.

H8 separated initialization/stochastic training, training membership and validation membership with disjoint reservoirs. It produced **91 unique fits**, not 105 independent fits: the three five-replica arms share seven anchor fits. Credit-g, Titanic, diabetes and ionosphere were **MIXED**; electricity and breast-w **STABLE**; blood-transfusion **INIT_DOMINATED**. Credit-g's normalized arm-variance shares were 54.9% init, 44.1% train and 1.0% validation. Diabetes had 48.6% validation; blood had 92.7% init.

These are conditional sensitivity measurements on one frozen outer split, not an additive causal variance decomposition. H8 effectively trained on about 39.2% of each dataset; changes in training membership also refit vocabulary/preprocessing. It refines, rather than contradicts, H7's broader split diagnosis. No common factor met the frozen rule, so **zero intervention or generalization fits** were run. Full-data refit, new validation fractions, ensembles and adaptive updates were not tested as H8 remedies.

## H9: AVG3 improved quality but failed robustness

AVG3 averages the three lowest-validation-loss checkpoints along the same trajectory, accumulating in float64 then restoring original dtype. It is neither three independently trained models nor inference ensembling. On 15 generalization development datasets, mean ΔAUC was +0.003182, median +0.000221, with 10/15 nonnegative. But variance ratio **1.484613** exceeded **1.05**. Amazon supplied 58.6% of net AUC gain and had variance ratio 4.641632; excluding it after the fact cannot rescue the registered result. Sonar improved locally. Keep BEST checkpoint in the product; **SPECIALIZED_CHECKPOINT_AVERAGING** is not approval for a K/EMA/SWA sweep.

## H10–H11: representation signal without a justified hybrid

H10 tested eight datasets × two fresh seeds × baseline/raw/matched token models. T2 used one 16-dimensional token per original feature, one two-head attention block and a count-matched concatenation readout. Mean ΔAUC +0.016397, median +0.008990, and 6/8 nonnegative looked promising. Diabetes at −0.011019 breached the frozen −0.01 worst-dataset bound. Scientific integrity passed, but timing was contaminated by another scientific workload; cost gates were unavailable, not passed. Quality failure alone stopped the study.

The no-attention ablation and Tier2 were **not run**. Therefore gains cannot be assigned specifically to attention rather than tokenization/readout/stopping. H11 reanalyzed all eight datasets and stopped offline: numeric-only tasks did not systematically deteriorate, and categorical fraction was not a sufficient mechanistic explanation. The proposed hybrid was **never built**. Its generalization is unknown; **NO_HYBRID_HYPOTHESIS_JUSTIFIED** is a decision not to start it.

## H12: a small INPUT_GATE candidate, not a product replacement

Fifteen canonical challengers were screened; EMA, CROSS2 and INPUT_GATE reached Stage 2, and only INPUT_GATE survived. The gate is identity-initialized contextual input scaling (context width 8) before the residual MLP. Full development covered 39 datasets (22 binary, nine multiclass, eight regression) and three fresh seeds. Raw mean NG +0.010461, median +0.003130, 29 W/5 L/5 T; capacity-controlled mean +0.011536. Isolated fit/prediction ratios were 1.111×/1.049×.

cpu_act contributed 37.8% of net gain. Removing the top contributor left positive mean +0.006678, below the strength threshold; the frozen concentration gate nevertheless passed. Multiclass uncertainty included zero. The static-scale ablation did not establish conditioning as the sole cause; all four tree families remained ahead in task means.

**NEW_BACKBONE_CANDIDATE_FOUND** is the exact historical verdict. It remains a modest research signal, not formally erased by later campaigns, but no integration or product promotion occurred. H13's INPUT_GATE reference and H18's missing counterfactuals do not provide a fresh compute-controlled promotion.

## H13–H14: BOOST quality replicated, mechanism advantage did not

BOOST freezes its width 64 learner, then trains a width 32 correction against the summed logits/standardized output and original labels.

Full H13: 39 datasets × three fresh seeds, mean NG +0.011627, median +0.005144, 31 W/8 L. All three task families met their practical rule, multiclass through macro-F1. Sonar lost −0.02197 AUC. A globally count-selected 52+32 control exceeded the budget on Bioresponse/Amazon and suffered a Titanic seed loss −0.15049. Thus **PROMISING_ALGORITHM_FOUND**, not a strong promoted backbone. AutoPrep stopped with zero median; adding INPUT_GATE was redundant under the registered comparison. Retrieval and Nyström retained modest regression-oriented signals with state/inference limitations.

H14 kept BOOST64+32 unchanged. Fresh confirmation passed quality: mean NG +0.010738841, median +0.005892256, 28 W/9 L/2 T. A widened single-MLP control matched parameters across all 39 datasets; BOOST retained advantage over it. The compute-controlled B0_LONG instead exceeded BOOST: BOOST−LONG mean **−0.007930**, median **−0.000661**, 14 W/22 L/3 T. LONG recovered **173.84%** of the BOOST-over-B0 mean gain and was better on each task mean.

The exact verdict is **BOOST_NOT_CONFIRMED**. Do not substitute `COMPUTE_DRIVEN`: that label's preregistered near-equivalence rule required absolute mean/median within 0.002, and the observed deficit was larger. The scientific lesson is that extra baseline compute explained or exceeded the apparent gain, while the formal decision remains distinct. Parameter matching and ablations did not override the failed compute mechanism gate.

## H15: stronger computation is not a universal new baseline

H15 compared complete20/25/30/35/40-epoch policies with patience 100 against public 30/4. C4=35 was selected globally before full outcomes. No tested point was formally saturated.

On 39 datasets × three fresh seeds, C4 produced mean NG +0.011244, median +0.002098 and 25 W/10 L/4 T. Regression mean ΔR² +0.018259 and median +0.014984 passed its practical gate. Binary median ΔAUC +0.000405 and multiclass practical thresholds failed, so **COMPUTE_SCALING_SIGNAL**, not general B_STRONG promotion. Isolated fit ratio 1.657719× belonged to that campaign, not every later use of the policy. Increasing epochs also retimed cosine and removed score-based early stopping; it did not isolate “more updates” as the sole cause.

## H16–H17: MIMO's concentrated mean did not justify safe routing

H16 screened17 forms with immediate budgeted baselines. Only MIMO had positive mean mechanism NG (+0.0151959), but median was −0.0103794, with 4 W/8 L. Titanic lost −0.0304321 AUC and triazines −0.1435025 R² against BC. Removing either car or sonar made the overall mean negative. No Stage B was triggered. WIDE_CROSSES had only 11 valid cost pairs and is not complete-panel evidence.

H17 performed no training. It examined both MIMO heads and the baseline on the 12 existing datasets. Harmful errors were strongly shared; on triazines even label-oracle head choice remained worse than B0. No universal observable safety signal met the frozen validation rule. Confidence-collapse associated with classification proper-loss harm (AUROC 0.7695 validation / 0.7618 test), but transferred weakly to correctness flips (test 0.5230) and supplied no regression coverage. SAFE was never constructed. **NO_SAFE_ROUTING_HYPOTHESIS** led to **NO_SAFE_SPECIALIST_SIGNAL**. MIMO is closed; no v2/v3/v4 rescue.

## H18: small quality headroom for the observed C0/C1 pool

C0 was public 30/4; C1 was H15's35/100. The paired primary cohort reused H15's39 datasets × three seeds, with a separate matched historical sensitivity cohort. Every leave-one-dataset-out global choice selected C1. Quality oracle-over-global mean NG **0.001350**, median **0**, and **7/39** datasets above0.002 failed both preregistered thresholds. No meta-policy, neural model or fresh seed was fitted.

The result is limited to this complete two-policy quality pool. Missing INPUT_GATE/recipe counterfactuals were not imputed. Utility-oracle headroom with the fixed compute penalty was **0.005765**, a separate unvalidated cost-oriented opportunity. The quality-first stop did not permit switching objectives after failure. A broad statement that “meta-learning cannot work” is unsupported.

## H19: bounded radical prototypes, including a preserved implementation failure

PC, tiny PFN-style prior-fitted context learning, relaxed logic and DEQ were implemented. The original PC used symmetric internal channels and its24 fits were marked **INVALID_IMPLEMENTATION_SYMMETRY**, retained as expenditure but excluded as quality evidence. A prospectively frozen asymmetric repair still failed: mean NG vs BC −0.371895, with near-constant predictions on some deep development circuits.

PFN had only 8/12 cost-eligible comparisons, negative eligible mean, and fully charged repeated synthetic pretraining. It was not a test of a large pretrained foundation model or amortized deployment. Logic failed the bounded screen. DEQ's binary Stage A hint advanced under an explicit specialist exception; fresh Stage B on18 datasets gave mean NG −0.055765 and median −0.028443, failing both general and binary-specialist rules. **NO_RADICAL_FAMILY_SIGNAL** applies to these prototypes, budgets and recipes, not to entire literatures.

## H20–H22: teacher quality and structural hints did not establish transfer

H20 used one fixed CatBoost300/depth 6 teacher recipe, three-fold training-only OOF predictions and an equal-weight true-label/teacher loss. The deployed student remained the public neural model. Against student-compute BC, mean NG +0.002121 but median −0.001376 and 5/12 nonnegative failed the primary gate. Defined teacher-transfer ratios had median 0. Only3/12 total-teacher-plus-student cost controls were fully funded; student-compute controls were12/12. Confidence weighting and leaf auxiliary supervision remained **NOT_TRIGGERED**, not negative fitted results.

H21 transferred soft split bases and tree co-occurrence pairs, with zero-initialized projection into the neural input. Stage A was broad against BC: mean NG +0.058229, median +0.036305, 10/12 nonnegative. But tree thresholds did not beat quantiles on average (T1−T2 −0.000267), so the complete primary gate failed. Tree graph versus random pairs was +0.008427 mean and zero median. Exact verdict **TREE_INTERACTION_SIGNAL**, qualified **DESCRIPTIVE_STAGE_A_ONLY**, candidate **NOT_PROMOTED**. No B, frontier or optional ablations ran.

H21's archived tree-export JSON files were overwritten by a Windows case-insensitive filename collision. Intact CBM files regenerated structures and predictions exactly; original JSON bytes were not recoverable. The archive discloses recovery rather than pretending those bytes survived.

H22 tested a distinct graph-only confirmation: generic basis for every feature, no tree thresholds, matched random/no-graph/generic-association controls. Eighteen datasets × two fresh seeds included six H21 overlaps and 12 additional development datasets, not external held-outs. Mean NG vs GC **−0.020484**, median **−0.007202**, 6/18 nonnegative; both overlap and prospective groups failed. Random-graph contrast +0.001595 became −0.001073 without ionosphere. Stable topology alone was not useful prediction evidence. **NO_TREE_GRAPH_SIGNAL** closes the tested recipe; H21 was not a promoted winner retrospectively revoked.

## H23: release with the frozen backbone and policy

Eight training-only methods preserved baseline deployment computation: CUTMIX, MASK_CONSIST, RDROP, VAT, ADV_HIDDEN, JACOBIAN, STOCH_DEPTH and GRAD_CENTRAL. All failed the strictly positive median against dedicated TC controls on the15-dataset one-seed screen; several incurred severe quality losses. No B stage ran, and no component was promoted.

Separately, exact H15 C4 regression policy was confirmed on eight datasets × five fresh seeds: mean ΔR² +0.019427, median +0.015268, 7/8 nonnegative R² and positive seed mean in all five seeds. Relative-RMSE NG was nonnegative6/8. Median executed epochs were14 versus35. The isolated public-policy replay still cost **2.295031× CPU**, **2.252357× wall** and **0.998613× prediction**, failing the unchanged 1.8× fit bound. Quality is retained; release promotion is not.

Accounting:341 scientific fits plus132 timing replays; no additional independent quality evidence from retiming. Final verdict **RELEASE_WITH_CURRENT_BACKBONE** ends discovery for 0.3; it does not authorize H24 or automatically execute release/publication steps.

# Historical Micro-Lab

**LEGACY / HISTORICAL INTERNAL LINEAGE. Historical evidence only; do not statistically pool it with H1–H23.** Cycle1–5 and Micro-Lab Batch 1–6 used older source, training semantics, datasets and hardware. Micro-Lab ran on Intel Core i3-2100 CPU with one Torch/BLAS thread, Python3.12.10 and torch2.7.1+cpu; its old virtual environment was nonportable. Current H-series CPU measurements are not direct speedup comparisons.

The master report records **648 scientific fits**, **54 unique MICRO IDs** and **98 registry records**. All infrastructure/test training counts are **UNKNOWN / NOT RECONSTRUCTED**. Final decision: **MICROLAB_PAUSE_UNTIL_NEW_HARDWARE**, queue **EMPTY**, **NO STACK JUSTIFIED YET**. Its final watchlist contained GELU, update-frequency dynamics and ranking/checkpoint/calibration, subsequently investigated in the H-series.

| Historical direction | Correct bounded outcome |
| --- | --- |
| Numerical view routing / top8 | No transferable general signal; guarded development gain reversed on public evaluation |
| CountSketch / interaction discovery | Sparse axis-aligned synthetic specialty; public development concentrated in phoneme; old held-out selected0/21; typed/rotated failures |
| SCSR | Quality signal with **COST_FAILURE** (6.9445× fit versus3× bound) |
| ESCT-v1 | **QUALITY_FAILURE** and **COST_FAILURE**; false stops and probability harm |
| Scalar ZI-QPLE-WARP-4 | **NOT_SUPPORTED_IN_TESTED_FORM**; real mean AUC about+0.000138 |
| Hidden adapter | Original integrity failure **INCONCLUSIVE**; rerun **COST_FAILURE**, quality gates incomplete |
| GELU | Batch 2 **PROMISING**; Batch 3 general **INCONCLUSIVE**, local glass2/digits gains; not a historical default |
| Update granularity | Batch 64/256 quality/cost problems; Batch 5 **INCONCLUSIVE**; Batch 6 frequency/LR **DIAGNOSTIC_FINDING** |
| Ranking/checkpoint/calibration | Batch 4 MICRO-0049 **PROMISING_MECHANISM**; disjoint Batch 5 replica MICRO-0054 **DIAGNOSTIC_FINDING**; Batch 6 MICRO-0056 **CALIBRATION_STABILIZED** |
| Winsorization | Synthetic contamination protection; winsor2 general **REJECT** after real glass2 harm |
| Categorical frequency / hashing | Mostly **INCONCLUSIVE**, no adequate transferable real-category evidence |
| Whole-vector embedding dropout | Early signal did not replicate; removed from watchlist |
| Capacity claim from XOR | Existing MLP learned low-noise XOR; legacy XOR failure did not prove architectural incapacity |

The calibration result is narrow: bounded/regularized T passed guards in all seven calibration-flagged unstable cases, but positive scalar temperature cannot change canonical logit ranking. FULL/ACCUM identity with dropout disabled separated optimizer-step frequency from accumulation; the one inverse-update LR rule removed the microbatch AUC advantage on both eligible diagnostics. These findings guide controls, not deployment defaults.

The original ZIP is unavailable in the current checkout; H12 preserves selected source/report/config/prototype members and an extraction receipt. Its SHA256 is `f4dd84389e9effa2e46fa6784a10c9ea81bb9dae2a902b0946bfe429b53d955d`. That is an archive fingerprint, **not a Git commit**. The handoff names a final consolidation ref that is absent locally; the archive-recorded Batch 6 object is absent too. A separate migration manifest was not found in reachable Git history. Consequently the historical final freeze SHA is **UNKNOWN / NOT RECONSTRUCTED**; the partial transfer is documented, not silently treated as a complete recovered repository.

ARCHIVE_REFERENCE: `research/h12/audit_sources/microlab/MICROLAB_MASTER_REPORT.md`, `NEGATIVE_KNOWLEDGE.md`, `HANDOFF_MANIFEST.md` and `PROTOCOL.md` in that same directory; `research/h12/audit_sources/ZIP_MANIFEST.json`; `research/h12/audit_sources/cycles/RESEARCH_CYCLE_2_REPORT.md`. Cycle1/3/3 B/4/5 conclusions above rely on the consolidated negative ledger; the preserved original Cycle2 report was also inspected. They are not claimed as newly replayed historical experiments.

# Negative Knowledge — Do Not Repeat Exact Forms

**NOT_SUPPORTED_IN_TESTED_FORM does not mean universal impossibility.** This ledger prevents accidental duplicate work. A closed exact default, a cost failure, an inconclusive study and a proposal stopped before implementation are different evidence classes, preserved in the reason column. Reopening requires new evidence or a materially different mechanism and its own protocol; the listed conditions do not authorize execution.

| ID / mechanism | Tested form | Why not promoted | Can it ever be reopened? |
| --- | --- | --- | --- |
| NK01 — Numerical view routing (Cycle1/Micro-Lab) | Cycle1 marginal evidence routing and guarded top8 | Routed AUC about -0.001901; top8 development +0.011434 reversed to evaluation -0.038301, 4/5 harmed. | New selection mechanism with nested leakage-safe validation, not the same rule. |
| NK02 — Numerical ensembles (Cycle1/Micro-Lab) | Cycle1 scalar/piecewise ensemble | About +0.005165 AUC at 2.96× fit and 2.23× parameters; no distinctive core contribution. | Explicit costly engineering comparator only, or materially different efficiency mechanism. |
| NK03 — Interaction discovery (Cycle1-2) | Cycle1 top2 marginal products; Cycle2 FM, bilinear, slots, TensorSketch, rank1 CrossNet, NID | Synthetic basis dependence, weak general transfer, unstable shortlists; all-pair expansion harmed AUC about 0.044693. | Typed, rotation-robust selection with abstention and direct cost controls. |
| NK04 — CountSketch interaction selector (Cycle2) | Residual safe-k2 and adaptive-k | Public development AUC +0.002923 concentrated in phoneme; 0/21 old held-out selections; typed/rotated failures and weak Python latency. | New typing/rotation/scaling mechanism; fixed synthetic specialty is not a general default. |
| NK05 — SCSR full update schedule (Cycle3) | Frozen 256/512/1024-step budgets with warmup/cosine | Mean AUC +0.021019 but median fit 6.9445× exceeded3×; COST_FAILURE. | Distinct budget hypothesis with credible cost model. |
| NK06 — ESCT-v1 (Cycle3B) | Validation every 32 updates, floor 64, two-event plateau | Gain retention 0.6746, LL harm+0.0110, worst+0.0791; false stops and tail-cost failure. | Independent stopping mechanism and explicit probability/worst-cost guards. |
| NK07 — Scalar numerical warp (Cycle4) | ZI-QPLE-WARP-4 | Cheap/safe but real mean AUC +0.000138; 0/7 datasets reached+0.003. | Materially different representation; no extra training rescue. |
| NK08 — Hidden adapter (Cycle5) | Historical low-rank adapter | Original integrity failure INCONCLUSIVE; rerun prediction1.617279×>1.15 after 6 pairs; quality not completed. | Credible inference-cost redesign; never relabel incomplete quality as failure. |
| NK09 — Micro-Lab optimizer/loss tweaks (Micro-Lab) | Batch 64, warmup 5, faster second moment, clip 1; smoothing 2%, focal gamma 1, balanced weights | Small heterogeneous screens did not establish broad quality/probability gains. | Separate target problem and mechanism; no neighboring sweep as general default. |
| NK10 — Stopping floors (Micro-Lab) | Minimum 12 epochs/16 updates, patience 12 | Unconfirmed; 12-epoch and 16-update forms identical21/21 under full batch. | Explicitly separate epochs, steps, schedule and selected checkpoint. |
| NK11 — Always-on robust preprocessing (Micro-Lab) | Robust tanh, signed-log IQR, rank-Gaussian; winsor1/2% | SPECIALIZED_ONLY; winsor2 protected contaminated synthetic 3/3 but harmed glass2 3/3; general form rejected. | Prespecified heavy-tail/contamination problem and abstention; no always-on retry. |
| NK12 — Premature stacks (Micro-Lab/H13) | Winsor1+GELU; calibration+microbatch proposal | Winsor+GELU beat best singleton0/2; calibration+microbatch never executed. | Two independently supported complementary mechanisms plus factorial BASE/A/B/A+B. |
| NK13 — Residual/categorical tweaks (Micro-Lab) | Residual scale.5/learned, RMS, input noise/dropout, frequency changes, hashing, whole-embedding dropout | Local/synthetic or INCONCLUSIVE; log-cardinality and embedding dropout did not confirm. | Distinct real-data mechanism with representative categorical evidence. |
| NK14 — AUC checkpoint and temperature (Micro-Lab/H4) | Raw AUC; AUC+T; FREE-T on small calibration | Ranking/probability conflict; FREE-T breast seed 107 LL2.290028; H4 calibrated delta AUC only+0.000153. | New calibration-specific question with disjoint partitions; positive T cannot repair ranking. |
| NK15 — Microbatch general rule (Micro-Lab/H3) | 64/256 and more updates independent of LR | Cost/probability failures; H3 mean disappears without Phoneme/Bank; LR scaling changes signal. | Mechanistically distinct frequency/LR/optimizer intervention, equal exposure/work. |
| NK16 — GELU general default (H2/H5/H6/H6C/CORE_GELU) | Activation-only SiLU replacement | H5/H6C positive; fresh core mean/median nonpositive, worst credit-g -0.045873; original integrity gap. | New context/mechanism with complete preregistered evidence; do not rerun exact default claim. |
| NK17 — Gated activations and simple MLP changes (H5/H12) | H5 GEGLU raw/matched, postnorm, modalitynorm, interaction8, width 96, depth 3; H12 SwiGLU | H5 seven forms rejected: median, breadth, concentration or worst-case failures; SwiGLU mean/task harm. | New hypothesis addressing the observed failure; capacity and compute controls required. |
| NK18 — Patience as general stability fix (H7/H8) | 4 to 6 with unchanged 30-epoch horizon | H7 median AUC delta0; blood worsens; credit variance 7.55×; H8 no common factor. | New identified cause, not patience/fraction sweep or retrospective exclusion. |
| NK19 — Checkpoint AVG3 default (H9) | Arithmetic mean of top3 validation-loss checkpoints in one trajectory | Generalization variance ratio 1.484613>1.05 despite positive AUC; Amazon concentration. | Distinct variance-control hypothesis with independent confirmation; no K/EMA/SWA rescue sweep. |
| NK20 — Uniform token-attention default (H10) | H10 dimension16, one block, two heads, matched readout | T2 diabetes AUC -0.011019 breaches-.01; timing invalid; attention ablation not run. | Distinct representation mechanism with worst-case and isolated cost proof. |
| NK21 — Hybrid by categorical narrative (H11) | Numeric-preserving/categorical-interaction proposal after H10 | H11 offline hypothesis unsupported; hybrid never built, not empirically falsified. | New evidence localizing a mechanism before architecture fits. |
| NK22 — H12 nonwinning canonical forms (H12) | EMA, CROSS2, FILM, ODST, PROTOTYPE, LOOKAHEAD, BATCH_ENSEMBLE, SAM, HIGHWAY, SOFT_BINS, RBF, ADDITIVE, SOFT_SELECT | No complete multi-task promotion gate; EMA and CROSS2 failed Stage 2 task/robustness; others Stage 1. | Different bounded mechanism and immediate strong controls; directional specialist labels are not validation. |
| NK23 — AutoPrep as general default (H13) | H13 small-probe metadata-admitted P0-P9 selector; B2/B3 | Stage 2 mean NG+0.027916, median 0; selection costly; B3 redundant with B2. | Distinct selection/cost objective and complete nested counterfactual evidence. |
| NK24 — Learned preprocessing (H13/H16) | H13 SOFT_TRANSFORM/MISSING_STATE; H16 transform mixer, basis bank, categorical router, context missing | No general finalist; SOFT_TRANSFORM zero-IQR floor produced extreme scale, so negative is implementation-bounded. | Numerical-safe materially different design, synthetic learning/sensitivity first, then BC. |
| NK25 — BOOST mechanism advantage (H13/H14) | Frozen 64+32 stagewise logit/residual correction with independent embeddings | Fresh quality replicated but BOOST-LONG NG -0.007930; compute control wins; no mechanism promotion. | New efficiency/mechanism argument beyond extra compute; no INPUT_GATE stack rescue. |
| NK26 — Universal B_STRONG/default35 epochs (H15/H23) | H15 C4 and H23 R1:35 epochs, patience 100, retimed cosine | H15 only regression practical; H23 regression quality passes but isolated fit CPU2.295031×>1.8. | New explicit quality/cost tradeoff or mechanism; preserve 30/patience 4 for 0.3. |
| NK27 — H13 other canonical operators (H13) | SPECTRAL, LOWRANK, HYPER, SCARF, MIXUP, DENOISE, ORTHOGONAL, RECURRENT, MAXOUT, SPLINE, CIN, MOE, TENSOR_TRAIN, MULTISCALE, SOFT_TREE, ADAGRAD, DEEPSETS | No strong full-suite mechanism; differing stop stages and early catastrophes; not whole-family disproof. | Consult exact ledger before proposing distinct operator; do not relabel screened forms untested. |
| NK28 — H16 other second-generation forms (H16) | FEATURE_MOE, TOKEN_MIXER, GRAPH_DIFFUSION, MASK_RECON_PRETRAIN, VICREG_PRETRAIN, METRIC_NCA, JOINT_RETRIEVAL, LATTICE_INTERACTIONS, WIDE_CROSSES, SEQUENTIAL_MASK, ADAPTIVE_DEPTH, LOWRANK_ENSEMBLE | No complete Stage A BC gate; WIDE_CROSSES11/12 eligible, not full-panel pass. | Different mechanism with numerical controls and valid compute coverage. |
| NK29 — MIMO / safe specialist rescue (H16/H17) | Two input-slot projections, shared representation, two heads; H17 routing diagnosis | Mean driven by car/sonar; negative median; Titanic/triazines catastrophes; shared errors defeat universal safety cues. | MIMO closed; no v2/v3/v4. New independent family/evidence, not a hindsight dataset router. |
| NK30 — C0/C1 quality meta-adaptation (H18) | Select public 30/4 vs35/100 using training metadata | Oracle-over-global0.001350, median 0, 7/39 above.002; no meta-policy fit. | Different justified pool/objective with sufficient headroom; cost utility is separate. |
| NK31 — Probabilistic circuit prototype (H19) | H19 corrected four-channel balanced conditional PC and frozen-sum control | Corrected mean NG vs BC -0.371895; near-constant predictions; original24 symmetric fits invalid. | Depth/input-sensitivity proof and canonical implementation audit before new real fits. |
| NK32 — PFN prototype (H19) | Tiny row-context model; 256 prior steps, 1024 synthetic tasks per fit | Only8/12 cost-valid cells; negative eligible mean; pretraining fully charged. | Different corpus/pretraining/amortization regime with full accounting; not a foundation-model rejection. |
| NK33 — Relaxed logic and DEQ (H19) | Bounded logic network; implicit64-state DEQ | Logic Stage A fails; DEQ binary hint fails fresh B, overall NG -0.055765. | Distinct optimization/representation proof; no universal family impossibility claim. |
| NK34 — Tree output distillation (H20) | CatBoost300/depth 6; 3-fold OOF; alpha.5, T1; unchanged student | Mean NG+.002121 but median-.001376, 5/12 nonnegative; median transfer ratio 0. | Different evidence-based transfer hypothesis, full teacher cost; confidence/leaf arms remain untested. |
| NK35 — Tree thresholds / graph transfer (H21/H22) | H21 soft basis then H22 generic basis+teacher co-occurrence graph | H21 quantile control fails; H22 NG vs GC-.020484 and random-graph benefit loses sign without ionosphere. | Different interaction evidence/representation, preregistered topology and generic-basis controls. |
| NK36 — Training-only release methods (H23) | CUTMIX, MASK_CONSIST, RDROP, VAT, ADV_HIDDEN, JACOBIAN, STOCH_DEPTH, GRAD_CENTRAL | All eight fail strictly positive median vs TC; no Stage B; several severe quality losses. | Distinct next-generation research question, not0.3 rescue or nearby setting sweep. |

# Positive Signals Not Promoted

These observations remain real within their stated scope. They are neither a hidden release queue nor independent confirmations of every historical result. A positive signal whose exact form is now closed remains useful diagnostic knowledge.

| ID / mechanism | Where it helped | Why not promoted | Reusable insight |
| --- | --- | --- | --- |
| PS01 — GELU historical | H2/H5/H6C real binary improvement; H5 AUC+0.004540 | Fresh core failure plus incomplete original integrity; NOT_SUPPORTED_AS_GENERAL_DEFAULT | Activation can affect small-task trajectories; no automatic default retry. |
| PS02 — Update-frequency dynamics | Micro-Lab XOR/digits; H3 Phoneme/Bank | Real generality fails after two best datasets removed; LR confounding | Use accumulation/equal exposure and explicit schedule controls. |
| PS03 — Robust calibration | Micro-Lab C2/C3 guard all 7 calibration-flagged unstable cases | No real ranking gain; H4 general component rejected | Calibration-only safeguards, separate from ranking, may inform a distinct objective. |
| PS04 — AVG3 | H9 Sonar/Amazon; general mean AUC+0.003182 | Variance gate fails, Amazon dominates | Checkpoint geometry/diversity diagnostics; no product averaging. |
| PS05 — Feature-token representation | H10 T2 mean AUC+0.016397; Titanic/Amazon/ionosphere | Diabetes worst-case failure; attention-specific cause not isolated; H11 stops | Preserve heterogeneous representation effects without categorical-only causal narrative. |
| PS06 — INPUT_GATE | H12 full 39:mean NG+0.010461, median+.003130; capacity-controlled gates pass | Modest effect, cpu_act37.8%net contribution; no core integration or new compute-matched promotion | Research-only candidate remains historically positive, not formally closed; must face strong compute baseline. |
| PS07 — BOOST | H13 mean NG+.011627; H14 fresh+.010739 vs cheap B0 | B0_LONG exceeds average advantage; BOOST_NOT_CONFIRMED | Stage 2 changes predictions usefully, but extra compute is a mandatory explanation/control. |
| PS08 — Regression compute scaling | H15 R2+.018259; H23 fresh R2+.019427, 7/8 nonnegative | H15 not multi-task general; H23 fit CPU2.295031× fails1.8 | Task-specific quality/cost frontier worth preserving for separately authorized 0.4 research. |
| PS09 — Retrieval and Nystrom correction | H13 regression mean R2+.015764 / +.009594 | Not two practical tasks; retrieval high inference/state cost; incomplete cost proof for strong final gates | Local residual information useful in bounded tasks; stored data and query cost must be counted. |
| PS10 — MIMO screen | H16 car and sonar; mean NG vs BC+.015196 | Median-.010379, 4 W/8 L, catastrophes; H17 no safe routing | Diagnostic example of concentrated mean and shared error; MIMO closed. |
| PS11 — Cost-oriented oracle headroom | H18 utility oracle-over-global+.005765 | Quality-first gate stopped; no cost policy learned or validated | Distinct cost-aware objective possible only under new preregistration. |
| PS12 — DEQ binary hint | H19 Stage A binary mean NG+.017116 | Stage B specialist median/breadth/worst gates fail | Bounded optimization evidence only; no specialist deployment claim. |
| PS13 — Tree structural discovery | H21 vs BC+.058229; vs random graph+.008427 | Threshold control fails; H22 prospective mechanism fails | Need isolate basis, feature allocation, thresholds, topology and compute; no confirmed graph recipe. |
| PS14 — CountSketch sparse interactions | Cycle2 axis-aligned synthetic sparse interactions | Typed/rotated and public transfer failures; historical only | Selection vs capacity distinction and memory/latency accounting. |
| PS15 — Classification harm diagnostics | H17 confidence-collapse loss AUROC.7695 validation/.7618 test | Correctness-flip test AUROC.5230; no regression coverage; not universal safety | Proper-loss associations differ from safe correction decisions. |

# Failure / Diagnostic Dataset Map

Observed associations and failures only: no causal diagnosis is inferred from dataset names, categorical fractions or a post-hoc favorable seed. Dataset identity and task must travel together; binary diabetes and regression diabetes are different datasets.

| Dataset / task / identity | Recurring failure | Sensitivity | Helped or hurt | Caution |
| --- | --- | --- | --- | --- |
| credit-g / binary / OpenML 31 | Product AUC.7020; fresh GELU mean delta-.045873; H23 several harmful training methods | H7 split-dominated; H8 MIXED init/train 54.9%/44.1%, validation 1.0% variance shares | H10 T2 mean+.011964 but one seed negative; H7 patience gain with variance 7.55×; H22 below GC | No missing/unseen pattern explains it generally; H8 shares conditional, not causal decomposition. |
| Titanic / binary / OpenML 40945 | Product AUC.7630; MIMO delta AUC vs BC-.030432; H13 matched BOOST worst seed-.15049 | H8 MIXED; H14 crossed probe DATA_SPLIT_INSTABILITY; categorical unseen issue observed | H10 T2 mean+.060000; H21 discovery gain; H22 small vs GC gain but not tree topology | Frozen 9 features; boat/body leakage fields excluded; schema/rare vocab and split/init confounded. |
| electricity / binary / OpenML 151 | Large tree gap despite stable short-run neural behavior | H8 STABLE; max arm SD about.0012 | H10 T2 AUC+.006015; H7 patience same predictions | IIDsplit is not a forecasting/time-series generalization experiment. |
| diabetes_binary / binary / OpenML 37 | H5 GELU small negative; H10 T2 mean AUC-.011019 breaches gate | H8 MIXED, validation 48.6% of normalized arm variances | H5 depth 3-.01685 AUC; H13 BOOST positive but not promotion | Different dataset from sklearn diabetes regression; zero values not silently reinterpreted as missing. |
| ionosphere / binary / OpenML 59 | Variable activation/representation effects; H22 graph contrast concentrated here | H8 MIXED; probe init/train SD ratio 1.97 does not pass2× dominance | GELU historical and H10 T2+.038667; H22 G1-G2 NG+.046957 | Single contributor can determine mean/sign; retain LL/Brier and leave-best-out. |
| blood-transfusion / binary / OpenML 1464 | H7 patience mean AUC-.009502924; longer budgets can hurt | H8 INIT_DOMINATED, init 92.7% normalized arm variance; validation unchanged | H3 microbatch selected loss; H14 BOOST negative; H15 binary worst | Variance reduction by harming difficult outcomes is not stabilization. |
| Amazon / binary / OpenML 4135 | H9 large AVG3 gain accompanied by variance increase | H9 AUC+.027990; variance 4.641632×, 58.6%net gain | H10 T2+.025976; H13 BOOST_MATCHED parameter 1.71501× due duplicated embeddings | IDs are categorical; capacity/cost accounting and rare/unseen behavior matter, not proof of cause. |
| sonar / binary / OpenML 40 | Small208-row problem; H13 BOOST mean AUC-.02197, variable later wins | H14 BOTH split and model-seed sensitivity; H22 edge Jaccard.066667 | Historical GELU/AVG3/MIMO wins; MIMO validation 0 corrections/2 breaks despite test7/0 | Large test win may lack validation routing cue; no stable specialist inference. |
| car / multiclass / OpenML 40975 | Productmajority-class collapse on2 seeds; rare-class weakness | Large budget and seed dependence; H23 VAT deltaaccuracy-.245665 vs TC | H16 MIMO/H21 basis apparentgains; H22 GC.894509 accuracy beats G1.778902 | Use active40975, not inactive21; improvement againstcheap B0 can vanishvscomputecontrol. |
| quake / regression / OpenML 209 | Product R2-.0140; allfourtreefamiliesalsonegative | Small or adverse gains across longertraining/tree methods | H23 R1 mean R2 delta-.0000555; H22 G1 below GC | Do not infer cause or labelno-signal universally from these splits. |
| triazines / regression / OpenML 206 | Product R2.0741, split-sensitive; MIMO delta R2 vs BC-.143503 | 186 rows60 features; H20 OOF relative RMSE range.675595-1.377691 | H20 distillationpartialgain; H23 R1 R2+.016297; H22 G1 worse | H17 label-oracle headchoice still worse than B0; shared errors, not merely averaging. |
| diabetes_regression / regression / sklearn bundled dataset | H13 BOOST errorworsening; H23 R1 R2 slightlypositivebut NG negative | Scalingmetrics can differ after seedaveraging | H22 GC betterthan G1; H23 R1 R2+.001222 and NG-.000491 | sklearn442-row dataset; never merge with binaryOpenML37. |
| cpu_act / regression / OpenML 197 | Trees ahead inproduct; large contributor to H12 gain | H12 INPUT_GATE37.8%net gain, leave-topmeanbelowstrengththreshold | H23 longertrainingpositive; H22 tinyvs GC gain | H12 sign survives deletion but promotion margin sensitive; keep contribution analysis. |
| auto_price / regression / OpenML 195 | Producttrees ahead; original-unit errorslarge | H22 individual losses; H23 RMSE mean dominated by scale | H15/H23 compute benefits | Never average raw RMSE/MAE acrosstargets as common effect size. |
| yeast / multiclass / OpenML 181 | Small imbalanced rare-class difficulties in product benchmark. | Class coverage and seed-dependent errors. | H21 threshold-basis discovery helped; H16 MIMO lost against BC. | No general rare-class repair established. |
| glass / multiclass / OpenML 41 | Small imbalanced rare-class difficulties in product benchmark. | H20 teacher OOF accuracy varied from0.534884 to 0.744186 across folds. | H20 distillation and H21 threshold basis showed local gains. | Do not confuse product multiclass glass with historical Micro-Lab glass2. |

# Current 0.3 Benchmark Context

These are the **frozen product** results, not the39-dataset research suite or a fresh held-out evaluation. There are **22 datasets and 330 fits:66 neural plus264 trees**. Paired60/20/20 splits, three seeds and five models per dataset. The original final report prints four decimals; the exact archived aggregate strings below follow `docs/BENCHMARK_0_3.md`. Independently recomputed release-audit figures differ only in the last floating-point digit for some metrics; this is not a scientific discrepancy.

| Task | Datasets × seeds / total fits | Exact neural aggregate |
| --- | --- | --- |
| Binary | 5 × 3 / 75 | AUC 0.8461200732305805; logloss 0.39058258431783444; accuracy 0.8117394824649958 |
| Multiclass | 9 × 3 / 135 | Accuracy 0.8275319528406876; macro-F1 0.7571351431249439; logloss 0.43709917510775587; macro-OVR AUC 0.9498761147868271 |
| Regression | 8 × 3 / 120 | R² 0.5320266908638694; relative RMSE 0.6273886356670959 |

Relative RMSE uses the split's **training-mean predictor** as denominator. Original-unit RMSE/MAE are retained per dataset, not averaged as comparable performance units. Their mean binary AUC spans0.8747–0.8892, multiclass accuracy0.8572–0.8703, and regression R²0.4783–0.5811. No universal ranking or SOTA claim.

Product binary weakness includes credit-g AUC0.7020 and Titanic0.7630; car has majority collapse on two seeds; yeast/glass retain rare-class difficulties. Quake R²−0.0140 and triazines0.0741 remain weak. cpu_act/auto_price favor trees in these comparisons; kin8nm favors the neural model here but is a public kinematics **simulation** benchmark. Timings include preprocessing/cold starts and were not fully machine-isolated; they cannot substantiate throughput or GPU claims.

Sources: `docs/BENCHMARK_0_3.md`, `release/0.3.0/RELEASE_AUDIT.md`. ARCHIVE_REFERENCE: `research/v0_3/FINAL_REPORT.md` and the binary/multiclass/regression benchmark reports alongside it. Binary refactoring preserved30 frozen reference cases through milestones A/B/B_fixed/C/D; no algorithm improvement is claimed from that equivalence.

# Compute and Fairness Lessons

**Future algorithm claims MUST include a compute-matched baseline from the first screen. Parameter matching is a separate requirement.**

- H14 BOOST improved cheap B0 but lost to B0_LONG: mean NG-0.007930; LONG recovered173.84% of the gain. Formal verdict is BOOST_NOT_CONFIRMED, not COMPUTE_DRIVEN, whose near-equivalence gate was not met.
- H15 proves a bounded compute-scaling signal, especially regression; more epochs also retime cosine and remove score-based stopping, so update-only causality is not identified.
- Future algorithm claims MUST compare with compute-matched baseline immediately; capacity matching addresses a separate confound.
- Report work proxy, achieved inclusive CPU, wall time, overshoot and unmatched cells separately; approximate dispatched work is not measured FLOP s.
- Count inner probes, teacher construction/OOF, pretraining, retrieval memory and inference. H20 total-cost controls were valid only 3/12; student-cost controls12/12.
- Separate training and inference cost. H23 unchanged inference graph did not make35-epoch training pass the1.8 xfit-cost gate.
- Isolated retiming may validate cost with exact prediction replay; it is not a new quality replication and contaminated attempts remain archived.

Timing repetition must specify scope: fit preprocessing, teacher/probe construction, validation, archive I/O, prediction transformation and cold start. Whole-process RSS is not model-only memory. Parameter count is not complete model state: retrieval buffers, kernel coefficients, support labels, fixed basis tensors and checkpoints matter. Include explicit unmatched cells and all correction attempts. Never choose the best-scoring BC attempt; cost-only allocation selects the registered eligible attempt.

H16–H23's >=90% work/CPU coverage is conservative approximate budget coverage, often with integer-epoch overshoot; it is not exact FLOP equality or proof of an optimal baseline frontier. H14's dense proxy omits costs such as embedding/validation work. H20 distinguishes student-compute and total teacher-plus-student questions. H21/H22 separately report teacher construction while their main neural controls cover student compute. Do not claim total-system efficiency from those neural-only controls.

# Preprocessing Knowledge

**P0 remains the product default.** TESTED records actual implementation/evaluation scope; NO GENERAL SIGNAL denies a general default, not every possible use. A recipe included in AutoPrep was evaluated by small inner probes, not necessarily an independently powered full-suite fixed-recipe study.

| Mechanism | Status | Tested form | Retained knowledge |
| --- | --- | --- | --- |
| P0 | TESTED | Current train-only median+standard, numeric missing indicators, category embeddings/log frequency, rare threshold2. | PRODUCT_DEFAULT; no H-series prep promoted. |
| Mean / neutral imputation | TESTED | H13 P1 mean and P2 zero with observed-statistics scaling admitted as selector recipes. | NO GENERAL SIGNAL for replacement; no independent causal superiority proved. |
| Missing states/indicators | TESTED | Baseline masks; H13 P9 explicit category missing masks and MISSING_STATE; H16 contextual missing. | Baseline retained; learned additions NO GENERAL SIGNAL. |
| One-hot / ordinal / hybrid | TESTED | H13 P4 full one-hot up to 256 columns; P5 normalized ordinal+mask; P6 one-hot cardinality<=16 otherwise embeddings. | Recipe selection evidence, not a complete independently powered fixed-recipe leaderboard. |
| Robust / quantile / rank | TESTED | Micro-Lab robust/signed-log/rank/winsor; H13 P3 Gaussian empirical rank and P7 IQR+frequency-only. | SPECIALIZED historical signals; NO GENERAL SIGNAL as default. Zero-IQR numerical safety matters. |
| Rare categories / frequency / embedding dropout | TESTED | Micro-Lab tweaks; H13 P9 sqrt(n)-based threshold and categorical missing; product reserved IDs and frequency. | Historical inconclusive/failed replication; baseline remains unchanged. |
| Target encoding | TESTED | H13 P8 threefold cross-fitted smoothed target statistics, smoothing 10; fold-complement labels/priors/normalization; full-train inference map. | Not UNTESTED; only 3 recorded B2 selected recipes, not general superiority. |
| AutoPrep | NO GENERAL SIGNAL | At most4 metadata-admitted recipes; 2 inner-fold hidden 32 probes, 4 epochs; fresh final model; selector cost included. | Stage 2 median NG 0; B3 AutoPrep+INPUT_GATE redundant. |
| Learned transforms | NO GENERAL SIGNAL | H13 soft transform numerical failure; H16 safe transform mixer, basis bank, categorical router, context missing all screened. | Negative tested forms; categorical router cheap B0 gain fails vs BC. |
| Unexecuted extensions | UNTESTED | H5 categorical RMS alternative; H18 learned selection over a complete broader recipe/INPUT_GATE pool. | Do not mistake proposals or unavailable counterfactuals for completed experiments. |

H13's selected B2 recipes across executed stages were P0:67, P3:40, P4:24, P6:6, P5:5, P8:3 and P9:2. These counts describe selector choices, not causal rankings of imputation or encoding quality. B2 and B3 reused identical probes and are not independent selector replications. Stage 1/2 mean pairwise choice agreement was0.778/0.717; this is conditional on a changing panel and recipe eligibility.

The soft-transform catastrophe included nonconstant zero-IQR features divided by a1e-8 floor, with transformed values up to 9e8. This is a numerical limitation of that implementation; later H16 safe transform/basis forms were materially different but still failed versus compute controls. Target encoding was indeed tested with fold-complement priors and train-only inference mappings. It must not be proposed as “never tested” in a future campaign.

# Algorithm Family Ledger

“Tested” means at least the named bounded local implementation. It does not establish full coverage of the family, faithful reproduction of every published method or a mathematical impossibility result.

| Family | Tested? | Best observed result / scope | Final status |
| --- | --- | --- | --- |
| Residual MLP | TESTED | Shared 64×2 SiLU product; H15/H23 regression compute gains | PRODUCT_DEFAULT; width/depth and general 35-epoch promotion failed |
| Gated MLP | TESTED | H12 INPUT_GATE small positive full-suite signal | RESEARCH_ONLY; not identical to inherited public feature_gating |
| GEGLU / SwiGLU | TESTED | H5 matched GEGLU mean+.000306, negative median; H12 SwiGLU heterogeneous | NOT_SUPPORTED_IN_TESTED_FORMS |
| Feature token / attention / mixer | TESTED | H10 positive mean, diabetes worst-case failure; H16 TOKEN_MIXER negative vs BC | NOT_SUPPORTED; H11 hybrid UNTESTED |
| Cross networks | TESTED | Cycle2 rank-1 CrossNet negative; H12 CROSS2 regression-oriented | No general promotion; different exact forms |
| Factorization / polynomial / tensor | TESTED | Cycle2 FM/bilinear/TensorSketch; H5 interaction8; H13 CIN/tensor-train | No robust general transfer |
| Mixtures / ensembles | TESTED | Cycle1 costly ensemble; H12 BatchEnsemble; H13 MOE; H16 Feature MOE/low-rank ensemble | No promoted mixture or ensemble |
| Differentiable trees | TESTED | H12 ODST and H13 SOFT_TREE screens | No general signal; not all tree methods disproved |
| Retrieval / prototypes / kernel | TESTED | H13 retrieval/Nystrom regression signal; H12 prototype and H16 NCA/joint retrieval failed | SPECIALIZED_UNPROMOTED; state/inference cost counted |
| Probabilistic circuits | TESTED | H19 corrected four-channel PC learns small synthetic but weak real predictions | NO_RADICAL_FAMILY_SIGNAL; original symmetric PC invalid |
| DEQ | TESTED | H19 binary Stage A exception then fresh Stage B failure | No confirmed specialist |
| PFN prototype | TESTED | H19 tiny synthetic prior-fitted row-context prototype | Negative partially cost-eligible screen; large pretrained systems UNTESTED |
| Logic | TESTED | H19 relaxed Boolean circuit prototype | No signal in bounded form |
| Preprocessing adaptation | TESTED | H13 AutoPrep; H16 learned input; H18 C0/C1 oracle | No promoted adaptation; H18 meta-policy NOT_TRIGGERED |
| Self-supervised / training methods | TESTED | H12 EMA/SAM/Lookahead; H13 SCARF/denoise/mixup; H16 masked/VICReg; H23 eight forms | No general training-only promotion |
| Tree output distillation | TESTED | H20 alpha 0.5 OOF teacher; mean positive, median negative | No promotion; confidence/leaf auxiliary UNTESTED |
| Tree structure transfer | TESTED | H21 descriptive graph hint; H22 prospective failure | NO_TREE_GRAPH_SIGNAL for tested graph recipe |
| Boosting / stagewise neural | TESTED | H13/H14 quality positive against B0 | BOOST_NOT_CONFIRMED against compute control |
| MIMO | TESTED | H16 mean concentrated in car/sonar; H17 no safe routing | CLOSED; no MIMO v2/v3/v4 |

Related but unexecuted forms must stay distinct: H11 hybrid, H17 SAFE, H18 fitted meta-policies, H20 confidence-weighted and leaf-auxiliary students, and conditional H21/H22 ablations were stopped before fitting. Large amortized pretrained PFN systems are outside H19's tiny repeated-pretraining prototype.

# Seed Ledger

**PARTIAL_ROLE_AWARE_PRIMARY_LEDGER.** The table reconstructs executed primary model/split seeds from the named reports/manifests. Fresh confirmation means fresh relative to the named campaign's historical audit **at that time**; every such seed is now used. Fresh split/init seeds on reused datasets are not unseen-dataset confirmation. The JSON development union includes all executed primary seeds; its fresh-confirmation array is an intentionally overlapping labeled subset.

| Campaign | Role | Executed primary seeds |
| --- | --- | --- |
| Micro-Lab | DEVELOPMENT_USED | 101, 103, 107 |
| H1-H4 | DEVELOPMENT_USED | 17, 42, 2026 |
| H5/H6/H6C | DEVELOPMENT_USED | 17, 42, 2026, 73, 314 |
| CORE_GELU | FRESH_CONFIRMATION_USED | 211, 307, 401 |
| CORE_GELU_RECOVERY | DEVELOPMENT_USED | 17, 42, 73 |
| H7_A | DEVELOPMENT_USED | 17, 42, 73, 2026, 314 |
| H7_B | FRESH_CONFIRMATION_USED | 503, 607, 701 |
| H8 | DEVELOPMENT_USED | 17, 42, 73, 314, 2026 |
| H9_diagnostic | DEVELOPMENT_USED | 1601, 1709, 1801, 1907, 2003 |
| H9_generalization | FRESH_CONFIRMATION_USED | 2203, 2309, 2411 |
| H9_sanity | DEVELOPMENT_USED | 2801 |
| H10 | DEVELOPMENT_USED | 3109, 3203 |
| PRODUCT_binary | DEVELOPMENT_USED | 41, 137, 509 |
| PRODUCT_multi_reg | DEVELOPMENT_USED | 67, 193, 401 |
| H12_stage1 | DEVELOPMENT_USED | 5107 |
| H12_stage2 | FRESH_CONFIRMATION_USED | 6101, 6203, 6301 |
| H12_stage3 | FRESH_CONFIRMATION_USED | 7103, 7207, 7307 |
| H13_stage0_1_2 | DEVELOPMENT_USED | 130103, 131101, 131203, 132101, 132203, 132307 |
| H13_final | FRESH_CONFIRMATION_USED | 133101, 133203, 133307 |
| H14 | FRESH_CONFIRMATION_USED | 1401103, 1402203, 1403307 |
| H15_quick | DEVELOPMENT_USED | 1501103, 1502207 |
| H15_full | FRESH_CONFIRMATION_USED | 1513103, 1514207, 1515309 |
| H16 | DEVELOPMENT_USED | 2601103 |
| H19_A | DEVELOPMENT_USED | 1901103 |
| H19_B | FRESH_CONFIRMATION_USED | 1902207, 1903309 |
| H19_PC_repair | DEVELOPMENT_USED | 1904409 |
| H20 | DEVELOPMENT_USED | 2001103 |
| H21 | DEVELOPMENT_USED | 3101103 |
| H22 | FRESH_CONFIRMATION_USED | 4101103, 4122207 |
| H23_R | FRESH_CONFIRMATION_USED | 5101103, 5122207, 5143309, 5164103, 5185207 |
| H23_A | DEVELOPMENT_USED | 5206309 |

H11, H17 and H18 introduced no training seeds. H18 reuses H15 and H14 cohorts; H17 reuses H16. H6C reuses the H6 eligible seed identities; timing replays do not create fresh quality seeds. The product multiclass/regression seed 401 had already appeared in CORE GELU; do not call it globally unused.

**Planned, not executed training:** H8 generalization 1103/1201/1301; H10 Tier2 3301/3407/3511. H16 B/C/D/ablation, H19 C and PC-repair B/C, H20 B/C/probe, H21 B/C/diagnostic, H22 B/diagnostic and H23 B/C are similarly prospective-only; all verified arrays are in JSON `used_seeds.planned_not_executed`. Prepared split files are not fits. Avoid reusing even planned values until a new seed-role/collision audit checks derived values.

Primary seed sources are ARCHIVE_REFERENCE: `research/new_hardware/configs/campaign.json`; H5/H6C/CORE/H7–H10 final reports; `research/h12/SEED_AUDIT.json` through the applicable H23 audit files, including `research/h19/pc_repair/SEED_AUDIT.json`; product `research/v0_3/configs/*_suite.json`. H14 onward includes derived seed arrays for model offsets, splits, folds and randomizations. H12 bootstrap 9901 is auxiliary, not a fit seed. Historical scanner arrays mix used/planned values and can contain boolean artifacts; do not blindly flatten them into a consumed-seed list.

**HELD-OUT RESERVED: UNKNOWN / NOT RECONSTRUCTED; JSON null.** Historical exclusions61/73/89 are preserved separately, not asserted as a current reservation. The primary ledger contains69 distinct values; exhaustive historical derived/probe/synthetic/bootstrap/infrastructure consumption remains unknown. Before a new campaign, audit its complete proposed seed derivations against archived registries rather than choosing from apparent gaps here.

# Important Git Anchors

All SHA values in the following table resolve to local commit objects and were checked against the corresponding refs. They identify archived closure tips, not permission to merge research into the product. Full phase reports may have been written one commit before a final receipt; H23's final scientific commit is distinguished below.

| Ref / milestone | Verified commit |
| --- | --- |
| dev/0.3.0 | `7d2e00d4ee52c6289db3a1598b8589a87b03bd2a` |
| release/0.3.0 | `b3bda31ac86fd4a0bb011046615cc5bac79e1033` |
| research/core-gelu-candidate | `0245f14d46f70f658ddce2ecf1991b673a9c0b9f` |
| research/h10-feature-representation | `cf36ed51db090ebcd53b309faa5c4479e83ee566` |
| research/h11-hybrid-representation | `7767768e589bbcf1ca3dfac26383a9359aaa6679` |
| research/h12-omniscreen | `b44ec3043f46ae6a748efaed4b1431b3da6d9db9` |
| research/h13-grand-discovery | `6026fc17b1456eef241faf9dfdfb36f563fd977c` |
| research/h14-boost-hardening | `5b95bc25e74899f18577e797679ef8c68b8f32c0` |
| research/h15-compute-frontier | `920c2ffc91454bb48f4b5c408fcf4001d32661e8` |
| research/h16-second-generation | `8d12dec6bbc2a24aca9465b5553980ef4cfefcab` |
| research/h17-safe-specialist | `fbf9afbcb5157008431aa72b2510964196b717ab` |
| research/h18-meta-adaptive | `5ce2fac06c106f0b56052836ef488067769538ba` |
| research/h19-radical-families | `cd5945dbe895c52d108b20a660928c6d5b3fadb9` |
| research/h20-tree-guided | `410dba2269bf332f26a042953ea1c846fd42fb1b` |
| research/h21-tree-basis | `5aa4c4d36758c6e39f8d488659f9a436064280e5` |
| research/h22-tree-interaction-graph | `0c36290522f9facac0cfaaaaf171baac3bac5f50` |
| research/h23-release-performance | `60ae0ddda5f6a9b9651a9e2929cfe152676332ed` |
| research/h5-component-discovery | `937c3b8e5ef857f09726c9b8cd2d3792549aee89` |
| research/h6-gelu-promotion | `89de66de5ba7514aee2656f535378c062ca957a0` |
| research/h6c-gelu-clean-confirmation | `8302265f9c99e1e53f1e8107a00a0999765b8627` |
| research/h7-stability | `d2173ef7fcc46d738297430b015953e9a02bf88e` |
| research/h8-split-decomposition | `f8f6de842a769d7b9f3756fb8fbe86559226ee79` |
| research/h9-checkpoint-averaging | `4dfad11c5578dcfc05d52f64c9474695af8a6cd6` |
| research/new-hardware | `46b165f2630a85abed80370aa2d17e25a94cb4d9` |
| v0.2.0 | `5ad5c9474775cb2194aeb3ae9cc84c7ebb0d4e69` |

The release build source is `689779654e774a9237d101f9ed3c1987522399da`; the closing candidate adds release evidence. H23's scientific/archive commit is `94a63e64188d071b0e176a09aa98198fd65cd862`; its final receipt branch tip is `60ae0ddda5f6a9b9651a9e2929cfe152676332ed`. Main at reconstruction is `3610bc147c88b64aa4a469657cb599c699d93bc1`.

Historical Micro-Lab final consolidation/freeze commit: **UNKNOWN / NOT RECONSTRUCTED**. The archival handoff is evidence of its history, but the named refs and old Git objects do not resolve here; no unverified SHA is promoted to this table.

# Open Research Questions

1. Can a neural mechanism improve quality and worst cases beyond a properly budgeted residual-MLP control, not merely beyond cheap B0?
2. What controlled stopping/schedule/compute change preserves the replicated regression benefit at acceptable measured training cost?
3. Which mechanisms explain shared catastrophic failures on small mixed or high-dimensional datasets without using dataset identity as a router?
4. Why do tree advantages fail to transfer through the tested OOF outputs or co-occurrence graphs, and what direct evidence could distinguish representation, supervision and optimization?
5. Does a complete, economically justified configuration pool have cost-oriented adaptation value beyond its best global policy? H18 tested only C0/C1 quality selection.
6. Can task-specific future backbones outperform a shared model under task-wise and worst-case gates? Current heterogeneous results do not establish a routing rule.
7. Large amortized prior-fitted learning and canonical deeper circuit implementations remain outside the tiny H19 prototypes; feasibility and full cost are unresolved, not positive signals.

These are evidence-grounded questions, not a claim that a new useful algorithm exists.

# Future Priorities

No experiments are started or queued by this memory. The following seven priorities require a new explicit task and a frozen scope; 0.3 discovery remains closed.

| Priority / question | Rationale and evidence | Do not repeat | Minimum next experiment / check |
| --- | --- | --- | --- |
| P0 — Preserve the frozen product and evidence boundary | 0.3 is a validated multi-task expansion, not a research-backbone promotion. Evidence: H23; release audit. | No H24 or automatic reopening for 0.3; no heldout-driven tuning. | None for this memory. Before any separately authorized publication, validate the exact release artifact through configured remote CI; physical CUDA claims require hardware evidence. |
| P0 — Predeclare a strong compute comparator | Apparent mechanism gains repeatedly disappear with more baseline work. Evidence: H14, H16, H22, H23. | Cheap B0-only promotion, hidden teacher cost or calling approximate coverage exact equality. | For a new justified mechanism, fixed small mixed-task panel with B0, cost-allocated baseline and candidate; count work, inclusive CPU, parameters, inference and worst cases before expansion. |
| P1 — Regression quality/cost policy | Regression compute benefit replicated but release cost failed. Evidence: H15, H23. | Exact35/100 as0.3 default or blind epoch/patience sweep. | One preregistered alternative justified by trajectories; 8 regression datasets with fresh seeds, controlled schedule/stopping contrast and isolated public-policy timing. |
| P1 — Failure robustness before broader architecture search | Credit-g, Titanic, sonar, car, triazines repeatedly undermine averages. Evidence: H7-H10, H16-H17, H22-H23. | Dataset-name routing, MIMO rescue or treating confidence as universal safety. | One causal hypothesis with crossed split/init controls, validation-only decisions, stable controls and frozen catastrophic-loss gates. |
| P1 — Resolve residual INPUT_GATE evidence only with new controls | H12 positive research signal remains unintegrated and not formally closed. Evidence: H12, H13, H18. | H12 threshold rerun as fresh discovery or automatic BOOST/EMA stack. | If separately selected for 0.4, exact gate plus static-scale and compute-matched baseline on preregistered fresh development splits; task-native effects, concentration and cost. |
| P2 — Cost-aware adaptation headroom first | H18 utility oracle gap exceeds quality headroom, but no policy evidence exists. Evidence: H18 utility gap0.005765; H13 AutoPrepzero median. | Switch H18 objective posthoc or impute absent recipe outcomes. | Freeze cost utility and complete small configuration matrix; first offline oracle bound, then dataset-level holdout only if new gate permits. |
| P2 — Audit genuinely different model regimes before fits | H19 prototype failures are not family-wide results; tree transfer remains unexplained. Evidence: H19-H22; Cycle2. | Renamed PC/PFN/logic/DEQ prototype, same distillation alpha or same tree graph. | Canonical implementation and synthetic depth/input-sensitivity/cost checks; proceed to one real-data hypothesis only when its structural difference and full cost are explicit. |

# Product Roadmap

| Milestone | Scope |
| --- | --- |
| **0.3.0 — current target** | Multi-task product expansion: binary, multiclass, regression; existing shared SiLU core and 30/patience 4 policy; local release freeze complete |
| **0.4.0 — next research generation** | Performance and/or new backbone work that beats properly budgeted controls with robust task-wise evidence; no candidate promised |
| **1.0 — maturity goal** | Stable API, mature validation and documented support boundaries |

No release dates are inferred. A release-candidate engineering verdict, research promotion and publication are separate decisions. The present user-authorized delivery is local only.

# Archive References and Known Gaps

**Full historical raw evidence is preserved in the archived research working directory.** ARCHIVE_REFERENCE entries below are repository-relative historical locators; they may not exist in the clean public copy. Use the verified phase ref only when a detailed archive audit is needed.

ARCHIVE_REFERENCE: H1–H4 share `research/new_hardware/reports/FINAL_REPORT.md`; H5–H23 use `research/hN/HN_FINAL_REPORT.md`, with N replaced by the phase number. The associated local branch and full closure commit are listed in Important Git Anchors.

H6C: `research/h6c/H6C_FINAL_REPORT.md`; CORE GELU: `research/core_gelu/FINAL_REPORT.md`. Their refs are in Important Git Anchors.

Supporting ARCHIVE_REFERENCE: H12 deep preregistration/leaderboards/extraction; H13 protocol and FAILURE_DATABASE/SIGNAL_DATABASE/VERDICT; H16 leaderboard/VERDICT/SIGNAL_DATABASE; H19 original/repair seed audits/H19_VERDICT; H21 H21_VERDICT; H23 PROTOCOL/INTEGRITY_AUDIT/FINAL_GIT_AUDIT. This reconstruction reads frozen audit claims, without reexecuting raw checkpoint audits.

Product sources are `src/neurotabular/_base.py`, `preprocessing.py`, `network.py`, the product final report, benchmark summary and release audit. Local delivery evidence is ARCHIVE_REFERENCE: `build/neurotabular-0.3.0-delivery/DELIVERY_README.md` and `DELIVERY_VERIFICATION.json` alongside it. Delivery ZIP s are filtered snapshots, not full Git-history backups. These two new memory documents do not rewrite or regenerate those frozen artifacts.

| Unknown field | Reason / consequence |
| --- | --- |
| H1_FORMAL_VERDICT | No standalone H1 verdict token; suite freeze documented. |
| HELDOUT_OPENML_IDS | Original six-dataset ID reservation manifest absent. |
| HELDOUT_RESERVED_SEEDS | Exclusion61/73/89 does not verify current reservation; 73was later used. |
| MICROLAB_FINAL_COMMIT | Old consolidation ref and Batch 6 object absent locally. |
| MIGRATION_MANIFEST | No separate migration manifest found; handoff/ZIP extraction receipt only. |
| COMPLETE_SEED_LEDGER | Primary executed seeds verified; exhaustive auxiliary/derived consumption unknown. |
| CORE_GELU_ORIGINAL_RAW | Original fresh CORE_GELU raw/source/split bindings absent. |
| MICROLAB_ALL_TRAINING_COUNT | 648scientific fits known; exhaustive infrastructure/test fit count unavailable. |

Bundled sklearn diabetes regression uses null OpenML identity. Unexecuted experiments are NOT_TRIGGERED/UNTESTED. Remote CI and physical CUDA are known pending statuses.

JSON schema_version1 declares compact tuple fields in `ledger_columns`; phases retain exact verdicts and verified commits. Null means not reconstructed, never zero.

