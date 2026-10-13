# NeuroTabular Research

Snapshot: **NeuroTabular 1.0.2**, 2026-10-12. Research is frozen after R38. This document summarizes historical evidence; it does not report a new experiment.

## 1. Research philosophy

NeuroTabular aims to be a practical, compact neural baseline for mixed tabular data. Complexity must earn its place through replicated quality, robustness and practical cost. A positive average alone is insufficient when worst cases, probability quality, variance, compute or integrity gates fail. Negative and inconclusive results remain useful knowledge.

Stable software contracts and exploratory algorithm work have different acceptance criteria. Research mechanisms are not silently enabled as defaults. Comparisons are workload-specific; the history does not establish universal state-of-the-art performance.

## 2. Research scope

The record spans neural architectures, numerical and categorical representations, interactions, regularization, optimization, checkpoints, ensembles, compute scaling, routing, MIMO, adaptation, tree-informed methods, probabilistic circuits, prior-fitted learning, deep equilibrium models, Bayesian approximations, hybrids and implementation performance. Small prototypes do not stand in for full scientific families or large pretrained systems.

In H19, “PC” denotes probabilistic circuits. A completed predictive-coding evaluation is not established by the reviewed history and is not counted as a tested family.

## 3. Research chronology

Historical identifiers are retained for orientation. Early phases precede the stable release. H24–H26 were recorded in early October 2026, H27–H33 and R34 followed, and R35/R36/D37/R38 were recorded on October 9. Stable 1.0.x engineering followed; 1.0.2 is the current product baseline. Earlier exact dates are not inferred.

| Phase | Motivation and approach | Evaluation | Result and decision |
| --- | --- | --- | --- |
| Legacy cycles and Micro-Lab | Explore cheap changes before larger campaigns. Routing, interactions, schedules, transforms and adapters; six Micro-Lab batches. | Small real and synthetic diagnostics; 54 distinct Micro-Lab IDs and 648 recorded scientific fits, with baseline reuse. | Local signals, reversals, cost failures and inconclusive screens motivated stronger controls. Exploratory evidence; no automatic promotion. |
| H1–H4 | Establish binary baselines and test optimization, activation and calibration. Paired baseline, GELU, batch/exposure and checkpoint/calibration tests. | Initially 13 binary datasets with fixed paired seeds; narrower trajectory tests for calibration. | GELU and update frequency showed local gains; no general calibration improvement. Advance selected hypotheses only to stronger comparisons. |
| H5–H6C; CORE_GELU | Confirm activation changes beyond short screens. Component tests, broader real panels and fresh confirmation. | Up to 22 eligible real datasets; H6C reused H6 results and is not an independent replication. | Earlier GELU gains were not sufficient for a general default; the fresh confirmation also lacked complete original integrity bindings. Keep the SiLU default; preserve the historical positive signal and integrity limitation. |
| H7–H9 | Reduce sensitivity to stopping, splits and trajectories. Patience, crossed seed-role diagnostics and checkpoint averaging. | Paired real binary panels with separate diagnostic and confirmation roles. | Patience did not establish general stability. Averaging improved mean AUC but failed variance and concentration gates. No general stability or averaging mechanism promoted. |
| H10–H11 | Test feature-token representations and their explanation. Uniform attention followed by offline causal review. | Eight datasets with paired seeds for attention; no new fits in the follow-up. | A positive average hid a worst-case failure; the proposed hybrid explanation was unsupported. No representation replacement; hybrid architecture was not built. |
| H12–H15 | Find a practical multitask backbone improvement. Input gating, preprocessing selection, stagewise learning and stronger compute controls. | Staged tests culminating in a 39-dataset development panel. | Input gating remained historically positive and unpromoted. Stagewise gains failed the stronger compute comparison; longer training showed regression-specific gains. No new stable default; retain scoped positive signals. |
| H16–H18 | Search for specialist or adaptive headroom. Seventeen bounded forms, MIMO failure analysis and policy-oracle analysis. | Small staged panels plus offline analysis of fixed policies. | MIMO had concentrated gains and severe failures; no safe universal cue or sufficient quality-first policy headroom emerged. No specialist routing or learned meta-policy. |
| H19 | Test bounded alternative computational families. Probabilistic circuits, prior-fitted prototypes, relaxed logic and deep equilibrium models. | Staged small real/synthetic tests with fresh confirmation for the DEQ hint. | Invalid circuit attempts were separated from corrected weak results; the DEQ hint did not confirm. No radical-family signal in the exact tested forms. |
| H20–H23 | Use tree knowledge or training methods to improve the backbone. Out-of-fold distillation, structural transfer and training-only methods. | Teacher-inclusive cost controls, prospective graph controls and fresh regression confirmation. | No broad practical survivor; an early structural hint did not pass its controls. Release with the current backbone. |
| H24–H26 | Test task-specific models, amortization and uncertainty. Eight task-specific forms, a shared synthetic-prior pilot and a last-layer Gaussian approximation. | Staged task panels, synthetic acceptance gates and ten regression datasets with three seeds. | No task-specific survivor; synthetic gates stopped amortization before real data; uncertainty breadth and coverage were insufficient. No new backbone, pretrained predictor or interval feature. |
| H27–H29 | Decompose failures and assess complementary models. Retrospective diagnostics, neural/tree blending and official external neural recipes. | 23 diagnostic cohorts, an 18-dataset blend panel and staged external comparisons. | Multiple partial failure mechanisms; blend cost/robustness failed; external results were mixed with incomplete matched coverage. No single explanation, hybrid promotion or automatic external adoption. |
| H30–H30C | Test adaptive capacity growth. Mechanical gate, repair, then controlled scientific comparison. | Synthetic mechanics followed by 15 observational datasets and two seeds. | Repair passed mechanics; the scientific comparison failed quality and robustness. Exact growth recipe closed. |
| H31–H33 | Identify and validate a justified next mechanism. Prior-art/control review, variable-projection mechanics and candidate red-teaming. | Algebraic fixtures and analysis; no scientific variable-projection fits. | Numerical acceptance failed; no subsequent candidate met admission requirements. Quality unevaluated for variable projection; no next mechanism. |
| R34–R35 | Consolidate knowledge and seek new reopening evidence. Source-bound history review and literature audit. | Existing records and reviewed papers; no new scientific fits. | New-to-corpus evidence was insufficiently actionable. Research remains frozen; historical roadmap proposals are superseded. |
| R36 | Check failure hypotheses on newly catalogued development data. Grouped duplicate-feature splits, fixed neural/tree recipes and perturbation controls. | 27 UCI datasets, nine per task, three seeds; 306 fits with no failed, censored or retried fits. | Ten material neural losses and several counterexamples; causal hypotheses remained partial. No research reopening; freshness is relative to the audited catalog. |
| D37 | Make future research auditable. Identity, alias, seed, overlap and failure-panel infrastructure. | 117 scoped identities and 80 campaign records; prospective failure panels were not executed. | Useful navigation and heuristic checks; incomplete first-use and reservation evidence remain. Infrastructure readiness is not research admission. |
| R38 | Decide whether evidence justifies another research phase. Zero-fit candidate review and inexpensive arithmetic checks. | No estimator fits, scientific solvers, dataset loading or new seed campaign. | No candidate met the combined scientific, mathematical and practical requirements. RESEARCH_REBOOT_NOT_JUSTIFIED; primary and backup NONE; held-outs CLOSED. |
| P35–P43X; 1.0–1.0.2 | Turn the existing model into a stable software baseline. Performance parity, API lifecycle, examples, persistence, compatibility and release engineering. | Bounded engineering and contract validation; separate from algorithm comparisons. | Stable contracts and measured implementation improvements without a new algorithmic promotion. NeuroTabular 1.0.2 remains the product baseline. |


## 4. Stable mechanisms

The stable defaults retain training-only numerical statistics, explicit categorical states and embeddings, category frequency, a compact residual SiLU backbone, ordinary classification/regression objectives and bounded validation-loss selection. These support practical mixed-data behavior. Lifecycle, schema and caller-ownership guarantees survived software contract and parity validation; they are not claims of superior predictive accuracy.

| Status | Mechanism | Retention rationale / boundary |
| --- | --- | --- |
| STABLE_DEFAULT | Train-only numeric representation | Median imputation, standardization and missing indicators; limits leakage and gives predictable input behavior. |
| STABLE_DEFAULT | Categorical states and embeddings | Training vocabulary, explicit missing/unknown/rare states and train-only bucket frequency. |
| STABLE_DEFAULT | Compact residual SiLU backbone | A bounded mixed-data neural baseline; research alternatives did not justify a general replacement. |
| STABLE_DEFAULT | Ordinary task objectives and bounded training | Binary/multiclass classification and single-output regression; validation-loss selection and bounded optimization. |
| STABLE_DEFAULT | Estimator lifecycle and caller ownership | Transactional fit state, schema alignment and protection of caller inputs and RNG state are software contracts. |
| SUPPORTED_OPTIONAL | Full-data refit and declared stable options | Existing opt-in refit and supported categorical/class-weight settings; refit cost is additional and it is skipped with external evaluation data. |
| EXPERIMENTAL_PUBLIC | Alternative numerical embeddings and feature gating | Existing opt-in behavior remains experimental; public feature gating is distinct from H12 research input gating. |
| EXPERIMENTAL_PUBLIC | Classified diagnostic and routing formats | Support is limited to the documented experimental scope; importability alone is not a stable promise. |
| RESEARCH_ONLY | Historically positive partial signals | Activation, calibration, averaging, input-gating, retrieval, compute and uncertainty signals remain scoped and unpromoted. |
| REJECTED | Exact forms failing their gates | Closed tested recipes retain their quality, cost, integrity or mechanical reason; this is not a rejection of entire scientific families. |
| REMOVED | No stable 1.0 removal established | Never-promoted research is not described as a shipped feature that was later removed. |


## 5. Non-promoted research

The following summaries preserve both useful signals and the reasons promotion stopped. “Not promoted” includes heterogeneous quality, excess cost, incomplete integrity, mechanical failure and non-admission; these are different conclusions. Rejection applies to the exact tested form and comparison, rather than an entire family.

| Family / phase | Problem targeted | Evidence and decision |
| --- | --- | --- |
| Activations and gated MLPs (H2, H5, H6, H6C, CORE_GELU, H12) | Improve optimization and input selection without excessive capacity. | Earlier GELU gains did not survive general-default confirmation. H12 input gating was historically positive but remained unpromoted; it differs from the public opt-in gate. |
| Feature tokens, attention and mixers (H10, H11, H16) | Represent heterogeneous feature interactions. | Positive average attention results coexisted with a worst-case regression in classification performance; the attention-specific explanation was not isolated. Mixer variants lacked general support. |
| Cross, factorization and tensor interactions (Cycles 2, H5, H12, H13) | Capture sparse or higher-order interactions compactly. | Synthetic or local gains did not establish robust real-data transfer; no general interaction replacement was promoted. |
| Routing, mixtures and ensembles (Cycles 1, H12, H13, H16, H17) | Adapt representation or allocate specialists to rows. | Routing gains reversed on evaluation or incurred excess cost. Mixtures and ensembles did not establish a robust, practical general benefit. |
| Differentiable trees (H12, H13) | Learn structured decision boundaries within neural training. | Bounded differentiable-tree screens did not support promotion; this does not reject all tree methods. |
| Retrieval, prototypes and kernels (H12, H13, H16) | Use local residual information. | Some regression gains were retained as signals; inference cost, retained state and insufficient breadth prevented promotion. |
| Regularization, optimization and stopping (Micro-Lab, H3, H5, H7, H15, H23) | Change update frequency, loss, batch size, regularization or stopping. | Most screens were inconclusive or specialized. Later controls exposed learning-rate, exposure and stopping confounds; no general replacement emerged. |
| Checkpoint selection, averaging and calibration (Micro-Lab, H4, H9) | Improve ranking, probability quality or trajectory stability. | Averaging improved mean AUC but failed variance and concentration gates. Calibration safeguards stabilized narrow cases without general real-data ranking benefit. |
| Preprocessing adaptation and meta-adaptation (H13, H16, H18) | Select a representation or training policy from training evidence. | Selection overhead and weak broad gains prevented promotion. The policy oracle did not justify training a meta-policy; an oracle is not an implemented learner. |
| Training-only and self-supervised methods (H12, H13, H16, H23) | Improve a fixed backbone through auxiliary training. | Bounded screens produced no general survivor. Regression compute gains were insufficiently practical under the frozen cost gate. |
| Stagewise neural boosting (H13, H14) | Fit residual information in an additional stage. | Gains against the cheaper baseline did not survive comparison with the stronger compute control. |
| MIMO (H16, H17) | Share training across multiple input/output streams. | A positive mean was concentrated in few datasets, with a negative median and severe failures. Subsequent diagnostics did not justify safe routing. |
| Probabilistic circuits and relaxed logic (H19) | Test bounded alternatives to ordinary neural representations. | The original symmetric circuit implementation was invalid; the corrected circuit and relaxed-logic prototypes lacked a general signal. |
| Deep equilibrium models (H19) | Use an implicit learned representation. | An initial binary signal failed fresh specialist confirmation. Only the bounded tested recipes are covered. |
| Prior-fitted prototypes and amortized learning (H19, H25) | Learn a reusable predictor from synthetic tasks. | A small prototype lacked practical broad evidence. The later shared amortized pilot failed frozen synthetic gates before any real-data evaluation. Large pretrained systems were not evaluated by these tests. |
| Tree output distillation (H20) | Transfer a tree teacher through out-of-fold training targets. | Positive mean but negative median and inclusive teacher cost prevented promotion; additional conditional arms were not triggered. |
| Tree-derived interactions (H21, H22) | Transfer structural information instead of teacher predictions. | An early descriptive hint failed the threshold control; prospective graph transfer did not confirm the mechanism. |
| Task-specific backbones and heads (H24) | Use task-specific grouping, class representations or regression objectives. | Eight bounded forms produced no survivor. Some panels stopped early after severe failures; a robust-regression gain depended on one dataset. |
| Bayesian last-layer uncertainty (H26) | Improve uncertainty without changing the point predictor. | Some Gaussian likelihood gains occurred, but coverage and breadth gates failed. The mean prediction remained unchanged; no interval API resulted. |
| Neural/tree hybrids (H28) | Recover complementary errors through an out-of-fold blend. | The blend improved over the neural baseline, but did not robustly improve over the tree comparator and had high inclusive cost and unstable weights. |
| Progressive residual growth (H30, H30R, H30C) | Allocate model capacity during learning. | Mechanical defects were repaired before scientific testing; the repaired recipe then failed quality and robustness gates. |
| Variable projection (H31, H32, H32R, H32V2) | Profile a supervised readout during representation learning. | Mechanical and numerical acceptance contracts failed before scientific fits. Quality remains unevaluated; the exact line is operationally closed. |
| Further admission proposals (H33, R35, R38) | Seek a justified next mechanism after earlier failures. | Overlap, incomplete specification, unsupported causal premises or practical scope prevented admission. Unexecuted proposals are not empirical negative results. |


## 6. External neural benchmark work

H29 compared fixed official TabM, RealMLP-TD-S, TabR and FT-Transformer recipes. Stage A used 15 datasets and two seeds; Stage B used 41 datasets and three seeds, with one simulator explicitly distinguished from observational data. Stage B completed 591 of 615 scheduled cells: 15 were censored, six timed out and three had attention incompatibilities. TabR did not pass the Stage A breadth requirement.

Against the cheaper N0, mean normalized gains were +0.050712 for TabM, +0.035154 for RealMLP and +0.038313 for FT-Transformer on their completed observational comparisons. Against the stronger compute control, eligible Stage B coverage was 102 of 120 observational cases for TabM, 118 for RealMLP and 100 for FT-Transformer. Their mean normalized gains were respectively +0.010940, −0.004906 and −0.003723. Mixed gains and incomplete coverage left all formal Stage B comparisons unsuccessful or incomplete: **INCONCLUSIVE_EXTERNAL_EVALUATION**. These normalized gains are not raw AUC differences.

Compute matching used phase-specific stronger neural controls, not equality of parameter count or a claim of identical FLOPs. Native recipe cost and excluded cells remained visible. On their recorded, differing coverages, median fit wall times were about 0.271 seconds for the neural baseline, 6.466 for RealMLP, 37.999 for TabM and 27.006 for FT-Transformer. Median hot batch-prediction latencies were respectively 4.436, 7.174, 52.834 and 56.845 milliseconds. These timings include the separate simulator supplement, whereas primary quality uses observational data. They are contextual engineering measurements, not universal speed ratios or single-row service throughput. Source pins, licenses and required adapters were reviewed; no external component was automatically promoted.

## 7. Failure decomposition

| Failure category | Evidence | Interpretation boundary |
| --- | --- | --- |
| Irrelevant-feature sensitivity | Perturbation and association evidence identified partial sensitivity, with confounding and counterexamples. | No general nuisance-removal cure was demonstrated. |
| Rare categorical support | Weak support, cardinality and conditional structure can coincide with failures; broad correlations changed after removing influential datasets. | Rare categories can also carry information; blanket suppression is not justified. |
| Sparse and conditional interactions | Interaction probes and tree comparisons exposed local representational gaps. | No controlled, practical mechanism solved them generally. |
| Small-data generalization | Some small datasets were difficult, but reduced-data controls contradicted a universal sample-size explanation. | Initialization, splits, support and finite capacity cannot be collapsed into one cause. |
| Target-tail behavior | Tail-related comparisons supported some contrasts but included clear counterexamples. | Neither a universal tail diagnosis nor a robust-regression default was established. |


H27 identified multiple partial mechanisms; R36 tested related hypotheses on newly catalogued development data. Neither demonstrated one cause or a mechanism solving all categories. A diagnostic association, row oracle or synthetic counterexample cannot by itself justify a deployable policy.

## 8. Benchmark methodology

Panels cover binary classification, multiclass classification and single-output regression. Protocols use fixed seed roles, paired data splits and training-only preprocessing. Nested selection and out-of-fold teachers use training data only. Later protocols group exact duplicate feature rows where specified. Reserved evaluation is separate from development.

| History | Design | Metrics and comparators | Limitations |
| --- | --- | --- | --- |
| Legacy cycles | Small public/development and historical internal panels; partial original manifests. | Binary ranking, probability quality and cost. | Exploratory; older held-out sets differ from current reservations. |
| Micro-Lab | Six batches, 54 IDs, 648 recorded fits; tiny real and synthetic panels. | AUC, probability losses, training/exposure and inference cost. | Baseline reuse, exploratory tiers and one-thread CPU scope; not independent 648-fold confirmation. |
| Public 0.2 engineering | Small public/synthetic workloads. | API, installation and performance measurements. | Engineering evidence; no broad accuracy ranking. |
| Public 0.3 quality | 22 datasets, three seeds, five fixed recipes: 330 fits; 5 binary, 9 multiclass, 8 regression. | Neural model and 100-iteration HGB, LightGBM, XGBoost and CatBoost recipes; no HPO. | Development panel; nonisolated timing and recipe-specific comparisons. |
| H1–H11 binary campaigns | Paired real panels growing from 13 to 22 datasets. | AUC plus log loss/Brier safety; phase-specific fixed seeds and controls. | H6/H6C reuse and incomplete original CORE_GELU integrity limit interpretation. |
| H12–H26 frontier | Staged multitask panels, including a 39-dataset development suite and narrower task panels. | Capacity/compute controls, task-native metrics and phase-specific normalized gains. | Early stops are not complete family rankings; teacher, selector and pretraining cost must be included. |
| H27/H28/H30C controlled work | 23 diagnostic cohorts; blend 18 datasets by two seeds; growth 15 by two seeds. | Failure associations, paired blend/growth quality and inclusive costs. | Retrospective reuse, confounded interventions, unstable blend weights and incomplete compute eligibility. |
| H29 external neural models | Official fixed TabM, RealMLP-TD-S, TabR and FT-Transformer recipes; staged panels. | Native quality, matched neural controls and inclusive fit/inference cost. | Incomplete matched coverage, censored attempts, timeouts and incompatibilities; no HPO or universal ranking. |
| R36 failure replication | 27 UCI datasets by three seeds; grouped duplicate-feature 60/20/20 splits, four CPU threads. | Fixed public 0.3.1 neural model versus HGB with 100 iterations, learning rate 0.1 and 31 leaves. | Fresh relative to audited catalog; observational findings do not prove a universal failure cause. |


Native metrics include AUC, accuracy, macro F1, log loss, Brier score, R², RMSE and MAE; uncertainty work adds likelihood and coverage. Ranking, probability quality and point error are separate objectives. Average seeds within datasets before panel summaries; report median, breadth, worst cases and leave-best-out sensitivity alongside means. Do not pool raw RMSE across target units.

Historical normalized gains use different definitions. H12 normalizes primary-score improvement by remaining baseline headroom with a floor. Later multitask campaigns use twice the AUC difference for binary tasks, chance-adjusted accuracy difference for multiclass tasks and favorable RMSE difference scaled by training-target standard deviation for regression. Product relative RMSE uses the test error of a training-mean predictor as denominator. These quantities cannot be pooled indiscriminately.

For context, the public 0.3 panel reported neural mean binary AUC 0.846120, multiclass accuracy 0.827532 and regression R² 0.532027. Fixed tree recipes ranged from 0.8747–0.8892 AUC, 0.8572–0.8703 multiclass accuracy and 0.4783–0.5811 regression R² on that workload. These historical aggregates describe a development panel, not current universal rankings or tuned competitors.

## 9. Known weaknesses

- Historical comparisons are workload-, recipe-, version- and budget-specific; they do not establish universal state-of-the-art performance.
- Tree baselines often outperform the neural baseline on important development tasks; no single mechanism explains all losses.
- Some positive means depend on a few datasets; worst-case, variance, cost and integrity gates can prevent promotion.
- Some original manifests, auxiliary RNG roles and all-in compute totals are incomplete; the original Micro-Lab container and CORE_GELU confirmation cannot be fully authenticated from available records.
- Dataset first-use and reserved held-out identity/seed records are incomplete; unknown identities are ineligible for reserved evaluation.
- External neural comparisons include missing or ineligible cells, timeouts and incompatibilities, so coverage and censoring must accompany comparisons.
- Synthetic learning, mechanical repair, algebraic checks and oracle analyses do not establish real-data generalization.
- Supported persistence is bounded by tested same-build and pinned-runtime contracts; universal cross-version portability is not promised.

## 10. Research governance

| Group | Rule |
| --- | --- |
| G01 — Authority | The latest decision governs; infrastructure and old plans do not authorize experiments. |
| G02 — Held-outs | Six reserved datasets remain CLOSED; unresolved reservation identities/seeds prevent opening. |
| G03 — Freshness | Resolve namespaces, aliases and lineage; new seeds are not new datasets and catalog absence is not proof of never-used status. |
| G04 — Training and splits | Keep preprocessing and selection training-only; pair splits and RNG roles and group duplicate features where required. |
| G05 — Experiment contract | Freeze hypotheses, panel, metrics, controls, seeds, budgets, attempts and stop rules before outcomes; mechanical gates come first. |
| G06 — Controls | Use strong compute and capacity controls; distinguish correlation, oracle headroom and causal evidence. |
| G07 — Aggregation | Average seeds within datasets; report mean, median, breadth, worst case, seed consistency and contributor concentration. |
| G08 — Metrics | Keep native metric meanings and phase-specific normalized-gain definitions separate. |
| G09 — Compute | Count selection, teachers, pretraining, solvers, refits, retrieval state, failed/censored attempts and cold/hot inference; missing cost is unknown. |
| G10 — Replay and failures | Preserve source-bound attempts; separate replay, retiming, replication, invalid, censored and unexecuted work. Parity cannot repair missing original evidence. |
| G11 — Reopening | Require a materially new premise, overlap and prior-art review, specification, controls, bounded cost and cheap falsification before admission. |
| G12 — Product boundary | Research is separate from stable defaults; maintenance is not an algorithmic quality claim. |


D37 added useful identity, seed, failure and overlap infrastructure, but its helpers are heuristic. The proposed 15-dataset failure panel and six-dataset fast panel were not executed; the proposed fast seed had already been used. First-use evidence and original reserved seed identities remain incomplete. Development success alone cannot open held-outs: independent replication, verified reservations, a frozen protocol and separate authorization are required.

## 11. R38 research freeze

Latest authoritative decision: **RESEARCH_REBOOT_NOT_JUSTIFIED**. Primary candidate: **NONE**. Backup candidate: **NONE**. Next scientific phase: **NONE**. Held-outs: **CLOSED**.

R38 found no candidate meeting the combined requirements for an identified failure, distinct mechanism, complete specification, practical cost and falsifiable controlled evaluation. It performed no scientific fits. Earlier promising labels, infrastructure readiness and historical roadmap suggestions do not override this decision.

## 12. Current research frontier

- How can statistical support constrain useful capacity without discarding rare informative features?
- Can sparse or conditional interactions be learned robustly under dependence and limited data at practical training and inference cost?
- How should irrelevant-feature sensitivity, rare-category behavior and target-tail failures be separated through controlled evidence?
- Support-adaptive sparse interactions remain an unresolved conceptual direction, without an admitted algorithm or demonstrated benefit.

These are open questions, not a validated algorithm, a claimed discovery or an experiment queue. Reopening requires the following sequence:

1. A zero-fit reopening audit establishes a materially new, source-supported premise.
2. An overlap check distinguishes the proposal from tested families and existing methods.
3. A prior-art and literature review clarifies established methods and unresolved questions.
4. An explicit mathematical and implementation specification defines targets, controls, support assumptions, cost and product compatibility.
5. A cheap falsification plan freezes mechanical checks, failure predictions, comparison rules, budgets and stop conditions.
6. An explicit admission decision must precede any possible scientific fits; held-outs remain CLOSED and require a separate opening decision.

## 13. Stable product relationship

Stable NeuroTabular 1.0.x remains insulated from failed and exploratory recipes. The classifier supports binary/multiclass tasks and probability prediction; the regressor supports single-output targets. Opt-in experimental representations remain explicitly experimental. No teacher, external neural model, retrieval system, uncertainty interval head, progressive growth or profiled-readout mechanism is silently substituted into stable defaults.

Performance and compatibility engineering can improve the product while leaving its algorithmic recipe unchanged. P35 reported median prediction improvements of about 30.7% and probability-prediction improvements of 35.8% on six CPU parity fixtures; those are bounded engineering results, not accuracy gains.

## 14. Reproducibility and evidence

This synthesis is based on reviewed historical reports, source-pinned configurations, result summaries and stable software contracts. Exact evidence is retained separately; raw internal evidence is not installed with the package. The companion RESEARCH_STATE.json records the same curated state. This public summary does not provide all raw artifacts required to independently reproduce every historical run.

Preservation does not repair an invalid experiment or fill missing original evidence. Replay, retiming, fresh confirmation, mechanical checks and unexecuted proposals remain distinct. All stated limitations accompany the historical conclusions; no new experiment or literature review was performed for this snapshot.

## 15. Current status

Product: **NeuroTabular 1.0.2**. Research: **frozen pending justified reopening**. Latest authority: **R38**. Primary and backup: **NONE**. Held-outs: **CLOSED**.
