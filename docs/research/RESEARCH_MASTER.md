# NeuroTabular Research

Scientific reconciliation, including post-C60 evidence stewardship: 27 October 2026. Product baseline: **NeuroTabular 2.0.0, published stable**, as confirmed by the maintainer. Latest completed neural research: **C60**. Subsequent archive recoverability work is reported separately below and is now closed. This documentation update changes neither the published artifacts nor the predictive implementation.

The mission is defined by the [canonical vision](../NEUROTABULAR_VISION.md). The [README](../../README.md) describes the available product. World-class accuracy and efficiency are goals, not an achieved superiority claim. The research remains open for future versions; release readiness did not close all scientific questions.

## Evidence and present decision

This is a curated public account of source-bound private reports, original protocols, retained results and the scientific index. It does not publish the private archive, reserved source identities, raw predictions or model checkpoints. Absence of a public raw payload limits independent reproduction from this document alone; it is not evidence that a result was never run. No new training or benchmark was executed for this reconciliation.

Keep six distinct questions separate: numerical correctness, functional equivalence, predictive quality, generalization, complete elapsed cost, and memory. Fewer parameters, lower gradient variance, smaller logits or fewer epochs alone answer none of the other questions. Temporary traced allocations and live tensor storage are not process peak RSS; RSS is not VRAM. Exploratory WSL measurements are not native performance certificates.

The current product retains five accepted engineering mechanisms from the original performance program. Its release does not integrate C60. The maintainer's documented release-policy deferral accepts residual unqualified performance risk; it does not turn missing Linux/Windows cost gates into PASS. Publication is not proof of a new general predictive, training-time or RAM improvement.

## Product history and integrated hardening

The published 1.0.3 maintenance baseline inherited the five accepted engineering mechanisms and release-identity safeguards. The local unpublished 1.0.4rc1 stage belongs to product stabilization, not a new published scientific baseline. The maintainer subsequently designated 2.0.0 as the stable maturity release; this milestone preserves the 1.x estimator API and defaults and introduces no new predictive architecture.

As recorded in the [changelog](../../CHANGELOG.md), 2.0.0 hardens rejection of complex numerical inputs beside nullable columns and complex sample weights before imaginary information can be discarded. It also rejects non-representable float64 conversions and invalid/overflowing training statistics explicitly while restoring the caller's NumPy error policy. Qualified ordinary conversions remain exact. Additional adversarial dtype, refit, persistence, dependency, private-file-exclusion and packaging checks improve reliability within their tested scope. These are correctness and release-quality changes, not general accuracy, calibration or speed gains. Earlier inconclusive guard-cost verdicts remain historical; the maintainer's later release-policy deferral does not certify that cost.

## Chronology and namespaces

The complete older H/Micro/P/R history, including R38, is preserved in the dated historical record below. R38 closed its own historical phase, not all subsequent NeuroTabular research. NP01-NP11 are prospective neural studies; WC01-WC19 are the first World-Class program; C01-C60 are the later post-1.0.3 series. E01-E08 below identify the eight earlier engineering cycles and must not be confused with C01-C08.

### Prospective neural program: NP01-NP11

| Cycle | Mechanism / question | Decision and verified scope |
| --- | --- | --- |
| NP01 | TabM-mini prospective neural screen | **REJECTED**. The fixed first prospective recipe completed 72 principal fits and 12 resource replicas on six new development cohorts, but failed its frozen predictive gates; no product promotion or general superiority. |
| NP02 | numeric rank channels | **REJECTED**. Raw and bounded training-rank channels did not meet the joint quality and inference-cost criteria. |
| NP03 | multiplicative input conditioning | **REJECTED**. The fixed low-cost conditioning recipe passed cost checks but not the required proper-loss benefit. |
| NP04 | initial gradient references | **REJECTED_PREMISE**. 196 gradient calls and no neural fit: the fixed-reference variance premise failed. This was not a test of every dynamic cache. |
| NP05 | induced feature attention | **REJECTED_COST**. The four-latent attention prototype passed mechanical checks but failed the CPU screen before predictive fits. |
| NP06 | stochastic gradient caches | **HETEROGENEOUS**. 5,012 gradient observations across 36 states showed relative variance 0.343, 1.005, 1.468 and 1.000 in four regimes. These are gradient-variance results, not accuracy gains. |
| NP07 | historical reliability and rare identity | **REJECTED_SCREEN**. Scalar cache reliability and an independent event-row embedding alternative failed their respective screens. A lost timing payload was excluded rather than reconstructed. |
| NP08 | numeric adaptation | **REJECTED_COST**. The fixed NCDA recipe failed the cost qualification; no predictive fit was admitted. |
| NP09 | data-conditioned initialization | **REJECTED**. The fixed initialization recipe was rejected after 69 completed neural fits; protected selection was not opened to rescue it. |
| NP10 | rank-one secant probe | **REJECTED_PREMISE**. 148 full-training gradient calls, no fits or updates: the single-probe premise failed; predictive quality was not measured. |
| NP11 | learnable ordered numeric intervals | **REJECTED_COST**. The numerical representation passed functional checks but failed the cost gate before predictive fits. |

### World-Class program: WC01-WC19

| Cycle | Mechanism / question | Decision and verified scope |
| --- | --- | --- |
| WC01 | competitive panel admission | **NOT_ADMITTED**. The initial competitive panel did not satisfy admission; the proposed 27 fits were not started. |
| WC02 | canonical competitive recovery | **SCOPED_EVIDENCE**. 24 principal fits, three fresh replicas and 600 paired inference windows established a canonical 32-member neural comparator on three development studies. Selection/preprocessing differences prevented attributing the gap to architecture alone. |
| WC03 | training recipe qualification | **NOT_ADMITTED**. The fixed panel was not admitted. Absence of fits did not falsify an optimizer or architecture. |
| WC04 | training budget and policies | **REJECTED_GENERAL**. No new Efficient or Flagship recipe passed all gates. The best paired median primary-metric improvement was 1.67%, below 2%; 144 derived policies were correlated evaluations, not independent experiments. |
| WC05 | representation and ensemble diagnosis | **DIAGNOSTIC**. Overfitting, a local benefit from more training, and a benefit from aggregating ensemble members were observed. No checked projection-rank collapse was found. Rare-category pooling lost identity, with predictive consequences unresolved. |
| WC06 | distillation economics | **CONDITIONAL**. A three-fold neural teacher-to-student recipe was technically plausible only as a reduced pilot. The full 72-fit plan exceeded the 27-fit budget; teachers were not treated as free. |
| WC07 | out-of-fold distillation pilot | **STOPPED**. Admission defects and an unfavorable local signal prevented promotion. This does not falsify neural distillation in general. |
| WC08 | distributional regression | **REJECTED**. The fixed eight-component CRPS regression recipe failed; component count and weights were not tuned after the result. |
| WC09 | regression failure diagnosis | **DIAGNOSTIC**. Fixed-checkpoint analysis distinguished mean bias and other possible limits without proving a universal representation or optimization cause. |
| WC10 | rare-category identity | **DIAGNOSTIC**. Pooling collisions alone did not establish predictive harm. Recurrence, support and a valid predictive contract were required. |
| WC11 | rare-category independent evidence | **NOT_ADMITTED**. Only one eligible family was available; the independent panel was insufficient. |
| WC12 | independent units and context | **HETEROGENEOUS**. Fourteen fits, including two qualification fits, showed heterogeneous unit/context effects; no general architectural causal conclusion followed. |
| WC13 | robust independent units | **NOT_ADMITTED**. Training composition and source availability limited admission. Group identifiers were required when the statistical contract required them, not automatically for every task. |
| WC14 | irrelevant-feature robustness | **REJECTED_CONTROL**. The fixed 27-fit screen failed its noise and positive-control gates despite a local real-data noise effect. |
| WC15 | signal and noise preservation | **SCOPED_POSITIVE**. A controlled synthetic noise effect passed local criteria in 21 fits. This was not independent evidence of a general real-data gain; the regularization follow-up was not admitted. |
| WC16 | independent signal robustness | **NOT_CONFIRMED**. The fixed 27-fit pilot did not pass the two-real-family gate. The earlier synthetic effect remained valid in its original scope. |
| WC17 | training-sample robustness | **NOT_CONFIRMED**. The fixed 27-fit pilot did not confirm uniform fragility and did not justify a selector. |
| WC18 | capacity allocation | **REJECTED_GENERAL**. The fixed 24-fit width-preservation recipe produced no robust material predictive benefit; incomplete CPU controls were retained as limitations. |
| WC19 | input sensitivity and interference | **DIAGNOSTIC**. Training-only fixed-model decomposition found no single dominant cause and admitted no architecture. Product stabilization remained a separate track. |

### Post-1.0.3 research: C01-C60

| Cycle | Mechanism / question | Decision and verified scope |
| --- | --- | --- |
| C01 | post-release recovery and panel | **COMPLETED**. Recovered baselines and scientific state; no predictive fits in this recovery cycle. |
| C02 | ordinal and count objectives | **REJECTED**. Twenty-four neural fits rejected the fixed ordinal-head and Quasi-Poisson recipes. Both remain closed. |
| C03 | first-projection conditioning | **REJECTED**. Eighteen neural fits and two convex controls gave an overall -0.14% result and failed the rotation control; no convincing speed or memory gain. |
| C04 | optimizer coordinate sensitivity | **DIAGNOSTIC**. Fixed-weight backward controls demonstrated Adam coordinate sensitivity. This was not evidence of a predictive improvement. |
| C05 | row-shared second moment and nullable path | **REJECTED_AND_PRELIMINARY**. The shared-row second-moment strategy was rejected. Nullable-path local timings were preliminary and required joint qualification. |
| C06 | nullable performance qualification | **REJECTED**. The original nullable recipe showed local gains but failed because of regressions and unstable measurements. Its verdict was not rewritten. |
| C07 | selective nullable path and complex guard | **INCONCLUSIVE**. Correctness evidence was favorable in scope, but cost qualification was incomplete; no integration. |
| C08 | measurement and conservative optimizer filter | **INCONCLUSIVE_AND_REJECTED**. Object-path A/A failed. A conservative Yogi-like filter failed a strong-rare-signal control; this did not falsify Yogi in general. |
| C09 | balanced feature dropout | **REJECTED**. The 27-fit recipe did not beat the frozen predictive controls. SMT interference remained a hypothesis, not an established cause of timing noise. |
| C10 | metrology and semantic premises | **DIAGNOSTIC**. Unreliable timing comparisons were suspended. Saved models and analytic controls tested semantic-loss premises without new fits. |
| C11 | semantic-aware representation | **NOT_ADMITTED**. Functional feasibility did not establish predictive admission or a generalization benefit. |
| C12 | semantic qualification | **REJECTED_COST**. The semantic recipe failed CPU qualification. Six baseline diagnostic fits did not justify promotion. |
| C13 | training dynamics | **DIAGNOSTIC**. Existing trajectories motivated checkpoint-selection questions, without establishing an improved learning algorithm. |
| C14 | guaranteed checkpoint selector | **REJECTED_PREMISE**. The selector premise failed before implementation or new fits. |
| C15 | sufficient-bound checkpoint selector | **REJECTED_SCREEN**. The fixed bound-based screen failed; thresholds were not changed to recover it. |
| C16 | scientific reset and source admission | **COMPLETED**. Reconciled evidence and qualified a restricted development panel without new learner fits. |
| C17 | causal generalization factorial | **SCOPED_EVIDENCE**. The development factorial was completed; independent binary confirmation and general promotion remained unavailable. |
| C18 | independent failure analysis | **SCOPED_EVIDENCE**. A restricted three-task development study supplied diagnostic evidence, not a promoted general recipe. |
| C19 | validation and failure diagnosis | **DIAGNOSTIC**. Validator and diagnostic controls passed in scope; a fresh independent binary source was not qualified. |
| C20 | data and confidence failures | **NOT_ADMITTED**. Source qualification stopped and confidence failures were identified; no new predictive fit was admitted. |
| C21 | numerical stability | **DIAGNOSTIC**. Stable-kernel checks on 100 extreme cases found no demonstrated numerical bug. Casting saturated probabilities to float64 cannot recover lost precision; NLL and clipped log loss have different semantics. |
| C22 | extreme-input provenance | **DIAGNOSTIC**. Extreme standardized values came from genuine out-of-support observations and sparse-feature imputation/variance effects. The stem and unnormalized residual propagated scale; float64 checks did not show a substantial numerical error. |
| C23 | invertible radial stem | **NOT_ADMITTED**. The transform passed numerical checks in its qualified domain; cost was acceptable only in the Flagship screen. Invertibility did not prove preservation of useful rare signals or predictive benefit. |
| C24 | robust numeric representation | **INCONCLUSIVE**. Training-observed-value scaling passed functional checks but A/A prevented cost qualification. Separate baseline/source diagnostics were not candidate efficacy evidence. |
| C25 | cross-group representation stability | **DIAGNOSTIC**. Fixed-model causal diagnostics did not justify promotion or independent predictive claims. |
| C26 | class coverage and training diversity | **NOT_ADMITTED**. The integrated runner was not yet qualified; a real predictive trial was not admitted. |
| C27 | runner and causal generalization | **NOT_ADMITTED**. Runner qualification remained incomplete. Data-integrity findings were retained without predictive promotion. |
| C28 | runner reliability and benchmark | **SCOPED_EVIDENCE**. The integrated runner was qualified in its checked scope; no broad generalization benefit was established. |
| C29 | convergence and frontend | **SCOPED_EVIDENCE**. Local training-budget and frontend diagnoses did not establish that more epochs generally improve prediction. |
| C30 | temporal information | **SCOPED_EVIDENCE**. Temporal signal was observed, without robust superiority of the proposed representation. |
| C31 | temporal order and RMS | **NOT_CONFIRMED**. A local order signal did not qualify general complementarity between RMS and lag information. |
| C32 | end-to-end efficiency | **CONDITIONAL**. Local cost checks were qualified, but independent validation of generalization remained blocked. |
| C33 | independent transfer | **REJECTED_GENERAL**. The fixed independent criteria rejected general transfer of the RMS-plus-lag recipe; it was not promoted. |
| C34 | probability and competitive audit | **DIAGNOSTIC**. A no-fit probability-quality and causal audit prioritized qualification of a competitive benchmark. |
| C35 | competitive benchmark qualification | **PARTIAL**. The software stage passed. Of 27 proposed trials, 21 were attempted: 20 passed, one stopped at its technical budget and six were cancelled. This was not an architectural verdict. |
| C36 | competitive scaling | **SCOPED_EVIDENCE**. Restricted competitive and resource evidence was collected. Different checkpoint metrics and budgets prevented direct causal attribution of the architecture gap. |
| C37 | training failure reconstruction | **DIAGNOSTIC**. All 34 baseline epochs were reconstructed: patience caused stopping and logged LR followed the scheduler. Ranking exceeded prevalence despite all-majority decisions at threshold 0.5. |
| C38 | stopping and scheduler causality | **SCOPED_POSITIVE**. Constant learning rate versus cosine had a positive result on two simulated development families under fixed capacity/work. This was not real-data confirmation. |
| C39 | independent training confirmation | **NOT_ADMITTED**. The independent real-data panel was not yet admissible; the original recipe and reservations were preserved. |
| C40 | independent dataset discovery | **NOT_ADMITTED**. Historical overlap, provenance and predictive-contract checks limited new-source admission. |
| C41 | independent dataset qualification | **NOT_ADMITTED**. Feature availability, rights and statistical-unit evidence still prevented admission; screening was not predictive evidence. |
| C42 | real-family predictive confirmation | **NOT_CONFIRMED**. The fixed C38 recipe was actually tested on two qualified real families. Uniform benefit was not confirmed: one family had a local 7.74% benefit. There was no default promotion. |
| C43 | training-dynamics decomposition | **DIAGNOSTIC**. Twelve retained trajectories and 360 epochs supported train/eval/dropout decomposition. No new fits were run and the proposed follow-up was stopped. |
| C44 | feature information limits | **DIAGNOSTIC**. Conflicting labels for identical permitted features imposed 69 unavoidable training errors in one deterministic contract. This was not a Bayes-error estimate or proof of an architectural limit. |
| C45 | information-aware learning | **NOT_ADMITTED**. Residual/local information diagnostics did not establish a sufficiently distinct new loss or admit new fits. |
| C46 | decision and representation diagnosis | **DIAGNOSTIC**. Ranking, threshold decisions and calibration were separated. No new threshold was selected on protected confirmation and no fit was promoted. |
| C47 | sequential validation views | **EXACT_SCOPED**. Technical equivalence checks supported private buffer/view changes; numerical equivalence alone did not qualify overall speed or RSS. |
| C48 | batch workspace and timing | **INCONCLUSIVE_TIMING**. Across the C47/C48 technical suite, 114 cases and nine fits supported equivalence. Only 2 of 64 timing A/A blocks were valid; IPC/acknowledgement failures and clock disagreement invalidated certification. |
| C49-C53 | grouped and tied representations | **REJECTED_AND_ARTIFACT_LOSS**. A 72-73% parameter reduction did not preserve predictive quality across development families. Twenty-seven model/artifact sets were lost through worker-temporary storage. Their absence is recorded; they were not recreated or replaced with invented artifacts. |
| C54 | durable experimental runner | **QUALIFIED_SCOPE**. Checksums, manifests, durable readback and interruption/resume were qualified for the tested CPU epoch-boundary path. This does not certify power-loss, GPU or arbitrary mid-batch recovery. |
| C55 | new grouped architecture | **REJECTED**. The new private architecture failed predictive criteria. Fewer parameters were not accepted as a substitute for accuracy. |
| C56 | early deep supervision | **REJECTED_GENERAL**. One local positive result did not generalize or pass the other control. Adaptive steps for non-interpolating objectives remained a separate question, not an admitted improvement. |
| C57 | eager CPU AdamW | **PROMISING_CONDITIONAL**. The constructor avoided 823 imports while preserving tested updates. Cold-start observations were exploratory; compiler and GPU paths were not fully qualified. |
| C58 | functional AdamW qualification | **EXACT_BUT_DEFERRED_COST**. 126 cases and 3,072 updates supported exactness in scope. First optimizer-state export imported the deferred stack; constructor savings were not a complete-fit speedup. |
| C59 | end-to-end and repeated costs | **CONDITIONAL_UNCERTIFIED**. Cold/no-state-export fits may benefit, but checkpoints can pay the deferred cost. Cache/layout alternatives lacked general benefit. One warm CPU profile assigned 80.22% to forward/loss/backward; it was not a universal profile or certified acceleration. |
| C60 | conservative allocation prototypes | **PRIVATE_NOT_INTEGRATED**. 24 software API fits, 48 initial/refit phases, 120 epochs and 48 fresh persistence loads supported tested equivalence. No independent predictive improvement, general timing gain or process-RSS reduction was certified. |

## What the training studies establish

C37 separated ranking from decisions at threshold 0.5 and reconstructed stopping without proving an architectural limitation. C38 then supplied causal development evidence for a constant-versus-cosine learning-rate contrast on simulated families. C39-C41 did not yet admit independent real confirmation. **C42 subsequently did run the fixed confirmation on two qualified real families and did not confirm a uniform benefit.** The local 7.74% result is not a general recipe. Later diagnostic and backlog statements must be read in light of that completed test, not used to repeat it under the same premise.

C43-C46 added information about train/eval behavior, feature information, label ambiguity, ranking, decisions and probability quality. They did not isolate a universal representational cause of the comparator gap. Different selection metrics, patience, schedules, budgets and update counts still prevent attribution to architecture alone. Data with identical permitted features and conflicting labels can limit deterministic training accuracy; that is not the unknown Bayes error on independent data.

## The eight initial performance-engineering cycles

The following are historical paired CPU engineering findings in their original workload scopes. The five accepted mechanisms were inherited by 1.0.3 and 2.0.0; they are not five newly demonstrated gains in 2.0.0.

| Engineering cycle | Mechanism | Decision / scope |
| --- | --- | --- |
| E01 | Categorical count dispatch | **ACCEPTED**. Preserved exact behavior in its qualified paired workloads. |
| E02 | Aligned string-schema fast path | **ACCEPTED**. Accepted schema fast path; an earlier missing-value guard attempt remained rejected. |
| E03 | Initial categorical mapper | **REJECTED**. Original fixed performance recipe failed; the historical verdict remains unchanged. |
| E04 | Float64 categorical mapper | **ACCEPTED**. Qualified under a new prospective protocol, without rewriting E03. |
| E05 | Private numeric buffer and C layout | **ACCEPTED**. Reduced unnecessary numeric-buffer work in the qualified scope. |
| E06 | Direct float64 concatenation and owned-output clipping | **ACCEPTED**. Accepted paired engineering change; streaming was not part of this cycle. |
| E07 | In-place multiclass normalization | **REJECTED**. Failed the joint gates despite local traced-allocation improvements. |
| E08 | In-place dropout, foreach AdamW and weighted-loss reuse | **REJECTED**. Fit ratios 0.98365, 0.98284 and 1.01247 did not meet 0.95; a fused alternative was not bit-exact on the first update. |

The 1.0.3 maintenance evidence included an exact paired median inference ratio of 0.85859636 (about 14.14% lower latency) in one retained multiclass probability workload, with six of six pairs faster. This arithmetic is workload-specific; it does not demonstrate universal fit, accuracy, RSS or generalization gains. The release program also checked package identity, installation, supported API behavior and persistence. Automated technical reviews and maintainer release decisions must not be described as a professional independent certification.

## Performance research, artifact integrity and C60

C47-C48 preserved numerical equivalence but failed timing qualification: only 2/64 A/A blocks were valid, with transport/acknowledgement failures and clock disagreement. Later RAW/MONOTONIC discrepancies were quantified without fully resolving the platform limitation. No repeated invalid campaign can replace an environmental gate. WSL, container, native Linux and native Windows evidence have different scopes.

The grouped architectures of C49-C53 reduced parameters by roughly 72-73% while degrading predictive quality on one development family. They were not promoted. Twenty-seven model/artifact sets were lost through temporary worker storage. Retained metrics and forensic evidence do not recreate missing models. C54 qualified durable artifact conservation and resume for the checked CPU epoch-boundary path; that qualification does not retroactively repair the loss or cover arbitrary crashes and GPU execution.

Eager AdamW in C57-C59 showed exact tested updates and avoided a constructor import stack, but a first optimizer-state export could pay the deferred cost. Cold processes, persistent processes, checkpointed fits and long fits must be distinguished. This remains conditional evidence, not a generally faster optimizer or a predictive improvement. A single warm CPU profile placing 80.22% in forward/loss/backward motivated attention to repeated neural work, without establishing a universal distribution.

C60 used real software execution: 24 API fits, 48 initial/refit phases, 120 epochs, 18 paired candidate fit cases, 48 fresh pickle/joblib loads, 54 inference bundle pairs, 216 edge comparisons across two dependency regimes, and 24 residual forward/gradient/RNG pairs. These are correctness/compatibility checks, **not an independent real-data generalization study or a certified timing/RSS benchmark**.

| Private C60 candidate | Result and limitation |
| --- | --- |
| Preallocated streaming predictions | **REJECTED_GENERAL_MEMORY_CLAIM**. An exact-output controlled workspace case increased live storage from 8.375 to 9 MiB. This is not trained-network RSS. |
| Shorter temporary-buffer lifetime before full-data refit | **PROMISING_UNCERTIFIED**. Two processed tables alive at refit entry fell from 11,088 to 0 bytes in the fixture; complete-fit time and peak RSS remain unqualified. |
| Regression inverse in an owned buffer | **CONDITIONAL_UNCERTIFIED**. A one-million-value float64 method trace fell from 9,000,289 to 1,000,193 bytes; small owned inputs had overhead. Views and read-only inputs retained a fallback. |
| Omit unused scalar quantiles | **PROMISING_UNCERTIFIED**. A 32,768 by 64 preprocessing trace fell from 67,395,160 to 50,352,060 bytes. Exact checked outputs do not certify full-process memory; private persistence state changed. |
| Reuse residual branch-add storage | **PROMISING_UNCERTIFIED**. Removed one 32 KiB allocation in a 128 by 64 float32 block fixture; 24 forward/gradient/RNG pairs and six fits were exact. No GEMM/FLOP reduction or significant speed benefit was proved. |

All five C60 prototypes remain outside 2.0.0 by explicit maintainer decision. Exactness applies to the checked cases and dependency regimes, not every supported device, hook, dtype or future dependency version. Allocation improvements in a method fixture are not enough to justify integration without significant end-to-end value, complete memory behavior, compatibility and maintainability.

## Post-C60 evidence stewardship and archive closure

The subsequent work concerned evidence preservation and recoverability, not a new neural research cycle. **C60 remains the latest completed neural campaign in this reconciled history.** No new predictive fit, timing certification or peak-memory experiment was performed for the archive audit or this documentation update.

The checked Scientific Atlas snapshot contains **2,593 nodes and 4,473 relations**. These are knowledge-map counts, not a count of independent experiments or evidence of a better algorithm. The existing 14 canonical indices, historical protocols, positive and negative results, source bindings and checkpoints were preserved. Raw archives and reconstruction dependencies remain private and are not included in this public account.

A stricter audit of 23 potential cleanup folders restored and SHA256-checked **390,450 regular files, totaling 30,262,878,535 bytes**, using 174 verified archive providers outside those folders. The audit separately retained 72 link/reparse payloads as metadata; it did not certify reconstructed link semantics, external Python bases or functioning restored environments. Byte recovery therefore did not imply eligibility for cleanup. The folder decisions were **2 PASS, 18 BLOCKED and 3 INCONCLUSIVE**. Protected project/baseline code, shared runtime dependencies and historical checkpoint bindings prevented cleanup where applicable. This audit covered currently present files; it did not recreate the previously lost C49-C53 model/artifact sets or missing NP07 timing evidence.

| Storage result | Recorded value | Interpretation |
| --- | --- | --- |
| Original storage actually reclaimed | **0 bytes** | No original was deleted or moved. |
| Potential file-data allocation in the two PASS folders | **576,405,504 bytes (549.70 MiB)** | Conservative potential only, not realized or authorized recovery. One MiB is 1,048,576 bytes. |
| Cleanup authority after campaign closure | **None, for all 23 folders** | The maintainer explicitly declined deletion, including the two PASS folders. |

The potential counts physical file identities once, excludes surviving external hardlinks and uncertain allocation categories, and does not add hypothetical directory/MFT, snapshot or recycle-bin savings. Earlier broader space estimates do not establish safely reclaimable capacity under this stricter audit. Additional audit evidence itself occupies storage, so the potential is not a net space saving already achieved. Only the generated restore-test copies were removed after verification; source originals, required providers and prior archives remain preserved.

**The space-recovery campaign is closed by maintainer decision.** No further cleanup, costly archive analysis or duplicate archival campaign is scheduled. The conditional C60 candidates and other open neural questions remain available for future, separately admitted work; campaign closure does not reject them or qualify their performance.

### Released product and documentation boundary

**NeuroTabular 2.0.0 is the published stable product**, as confirmed by the maintainer and consistent with the local version metadata. No account or registry was accessed to independently reconfirm publication in this update. The maintainer's release-specific performance deferral remains a documented decision, not a native-benchmark PASS or an independent certification. No unqualified C60 prototype was integrated and no general accuracy, training-speed, peak-RAM or VRAM improvement is established by publication or archiving.

This update changes only the two public research documents. Product code, defaults, version metadata, README, source-distribution manifest and the published wheel/sdist remain unchanged. These repository research documents are excluded from the distribution source manifest and sdist; updating them does not rebuild or amend released artifacts. Any subsequent GitHub upload remains a manual maintainer action.

## Open scientific frontier and stop rules

- Preserve promising conditional buffer-lifetime, regression-inverse, scalar-quantile and residual-storage evidence for future qualification. The general non-increasing-memory claim for streaming outputs is already contradicted by a controlled counterexample.
- Prioritize full learning work and time to a fixed predictive quality: forward/backward, convergence, batching and representation. Do not return to parameter reduction that sacrifices quality or repeat eager-import/cache/layout studies without a new premise.
- Study rare informative categories, irrelevant features, sparse and out-of-support numerical inputs, probability quality, training stability and independent generalization. A mathematically invertible transform can still lose usable information in finite precision; reducing logits is not better calibration by itself.
- Admit new sources from provenance, rights, lineage, feature availability, duplicates and the statistical unit. Group identifiers are essential when dependence requires them, but their absence is not an automatic veto for unrelated independent-unit tasks.
- Freeze data roles, splits, seeds, controls, appropriate predictive and calibration metrics, full compute/memory budgets and regression/stop rules before fitting. Qualify the real durable runner. Separate Efficient and Flagship criteria prospectively.
- Preserve reserved selection and historical held-outs, paired comparisons, negative results and incomplete evidence. Development data do not become independent confirmation by renaming or re-splitting them.
- For speed and process-memory claims, pass the frozen environmental and A/A controls before A/B. A platform failure is an infrastructure limit, not a demonstrated software defect. Fewer epochs are not a certified speedup.
- Stop unpromising fixed recipes promptly. Do not tune thresholds, seeds, temperatures or schedules after failure to recover the same hypothesis. New research remains possible without delaying an already frozen product indefinitely.

## Structured state schema

`RESEARCH_STATE.json` uses schema version `2.0`. Root fields distinguish current `product_version` / `product_status` from `latest_completed_research` / `research_status`. `chronology` contains stable campaign identifiers, mechanism families, scoped decisions, findings and integration flags. C49-C53 form one evidence-linked group; the identifier covers each of those cycles without inventing separate missing artifacts. `initial_engineering_cycles`, `integrated_engineering` and `non_integrated_c60` distinguish retained product mechanisms from research prototypes. `evidence_dimensions`, `reconciliations`, `admission_requirements`, `protections`, `open_opportunities` and `public_evidence` describe interpretation and governance. `historical_snapshots` retains the complete previous structured public history with its original date; its old current-state fields do not override the new root state. Consumers of the former schema must migrate deliberately rather than assuming identical keys.

`post_c60_evidence_stewardship`, `scientific_atlas_snapshot` and `public_documentation_update` are additive summary fields. They record archive closure, scoped recovery evidence and this documentation-only revision; they introduce no new neural campaign identifier and do not override preserved historical decisions.

Each chronology record is a public curated summary whose full source binding remains private. Decision labels describe a fixed recipe or tested scope; they do not falsify an entire algorithm family. Public source links identify admissible product documentation, not a claim that all private evidence is downloadable.

## Preserved public historical record through R38

The following complete earlier document is retained as a **dated historical snapshot**. References inside it to a current 1.0.2 product, a freeze after R38, old frontiers or then-missing evidence apply to that snapshot only. They do not describe the present product or erase NP, WC and C research above.

### NeuroTabular Research

Snapshot: **NeuroTabular 1.0.2**, 2026-10-12. Research is frozen after R38. This document summarizes historical evidence; it does not report a new experiment.

#### 1. Research philosophy

NeuroTabular aims to be a practical, compact neural baseline for mixed tabular data. Complexity must earn its place through replicated quality, robustness and practical cost. A positive average alone is insufficient when worst cases, probability quality, variance, compute or integrity gates fail. Negative and inconclusive results remain useful knowledge.

Stable software contracts and exploratory algorithm work have different acceptance criteria. Research mechanisms are not silently enabled as defaults. Comparisons are workload-specific; the history does not establish universal state-of-the-art performance.

#### 2. Research scope

The record spans neural architectures, numerical and categorical representations, interactions, regularization, optimization, checkpoints, ensembles, compute scaling, routing, MIMO, adaptation, tree-informed methods, probabilistic circuits, prior-fitted learning, deep equilibrium models, Bayesian approximations, hybrids and implementation performance. Small prototypes do not stand in for full scientific families or large pretrained systems.

In H19, “PC” denotes probabilistic circuits. A completed predictive-coding evaluation is not established by the reviewed history and is not counted as a tested family.

#### 3. Research chronology

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


#### 4. Stable mechanisms

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


#### 5. Non-promoted research

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


#### 6. External neural benchmark work

H29 compared fixed official TabM, RealMLP-TD-S, TabR and FT-Transformer recipes. Stage A used 15 datasets and two seeds; Stage B used 41 datasets and three seeds, with one simulator explicitly distinguished from observational data. Stage B completed 591 of 615 scheduled cells: 15 were censored, six timed out and three had attention incompatibilities. TabR did not pass the Stage A breadth requirement.

Against the cheaper N0, mean normalized gains were +0.050712 for TabM, +0.035154 for RealMLP and +0.038313 for FT-Transformer on their completed observational comparisons. Against the stronger compute control, eligible Stage B coverage was 102 of 120 observational cases for TabM, 118 for RealMLP and 100 for FT-Transformer. Their mean normalized gains were respectively +0.010940, −0.004906 and −0.003723. Mixed gains and incomplete coverage left all formal Stage B comparisons unsuccessful or incomplete: **INCONCLUSIVE_EXTERNAL_EVALUATION**. These normalized gains are not raw AUC differences.

Compute matching used phase-specific stronger neural controls, not equality of parameter count or a claim of identical FLOPs. Native recipe cost and excluded cells remained visible. On their recorded, differing coverages, median fit wall times were about 0.271 seconds for the neural baseline, 6.466 for RealMLP, 37.999 for TabM and 27.006 for FT-Transformer. Median hot batch-prediction latencies were respectively 4.436, 7.174, 52.834 and 56.845 milliseconds. These timings include the separate simulator supplement, whereas primary quality uses observational data. They are contextual engineering measurements, not universal speed ratios or single-row service throughput. Source pins, licenses and required adapters were reviewed; no external component was automatically promoted.

#### 7. Failure decomposition

| Failure category | Evidence | Interpretation boundary |
| --- | --- | --- |
| Irrelevant-feature sensitivity | Perturbation and association evidence identified partial sensitivity, with confounding and counterexamples. | No general nuisance-removal cure was demonstrated. |
| Rare categorical support | Weak support, cardinality and conditional structure can coincide with failures; broad correlations changed after removing influential datasets. | Rare categories can also carry information; blanket suppression is not justified. |
| Sparse and conditional interactions | Interaction probes and tree comparisons exposed local representational gaps. | No controlled, practical mechanism solved them generally. |
| Small-data generalization | Some small datasets were difficult, but reduced-data controls contradicted a universal sample-size explanation. | Initialization, splits, support and finite capacity cannot be collapsed into one cause. |
| Target-tail behavior | Tail-related comparisons supported some contrasts but included clear counterexamples. | Neither a universal tail diagnosis nor a robust-regression default was established. |


H27 identified multiple partial mechanisms; R36 tested related hypotheses on newly catalogued development data. Neither demonstrated one cause or a mechanism solving all categories. A diagnostic association, row oracle or synthetic counterexample cannot by itself justify a deployable policy.

#### 8. Benchmark methodology

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

#### 9. Known weaknesses

- Historical comparisons are workload-, recipe-, version- and budget-specific; they do not establish universal state-of-the-art performance.
- Tree baselines often outperform the neural baseline on important development tasks; no single mechanism explains all losses.
- Some positive means depend on a few datasets; worst-case, variance, cost and integrity gates can prevent promotion.
- Some original manifests, auxiliary RNG roles and all-in compute totals are incomplete; the original Micro-Lab container and CORE_GELU confirmation cannot be fully authenticated from available records.
- Dataset first-use and reserved held-out identity/seed records are incomplete; unknown identities are ineligible for reserved evaluation.
- External neural comparisons include missing or ineligible cells, timeouts and incompatibilities, so coverage and censoring must accompany comparisons.
- Synthetic learning, mechanical repair, algebraic checks and oracle analyses do not establish real-data generalization.
- Supported persistence is bounded by tested same-build and pinned-runtime contracts; universal cross-version portability is not promised.

#### 10. Research governance

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

#### 11. R38 research freeze

Latest authoritative decision: **RESEARCH_REBOOT_NOT_JUSTIFIED**. Primary candidate: **NONE**. Backup candidate: **NONE**. Next scientific phase: **NONE**. Held-outs: **CLOSED**.

R38 found no candidate meeting the combined requirements for an identified failure, distinct mechanism, complete specification, practical cost and falsifiable controlled evaluation. It performed no scientific fits. Earlier promising labels, infrastructure readiness and historical roadmap suggestions do not override this decision.

#### 12. Current research frontier

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

#### 13. Stable product relationship

Stable NeuroTabular 1.0.x remains insulated from failed and exploratory recipes. The classifier supports binary/multiclass tasks and probability prediction; the regressor supports single-output targets. Opt-in experimental representations remain explicitly experimental. No teacher, external neural model, retrieval system, uncertainty interval head, progressive growth or profiled-readout mechanism is silently substituted into stable defaults.

Performance and compatibility engineering can improve the product while leaving its algorithmic recipe unchanged. P35 reported median prediction improvements of about 30.7% and probability-prediction improvements of 35.8% on six CPU parity fixtures; those are bounded engineering results, not accuracy gains.

#### 14. Reproducibility and evidence

This synthesis is based on reviewed historical reports, source-pinned configurations, result summaries and stable software contracts. Exact evidence is retained separately; raw internal evidence is not installed with the package. The companion RESEARCH_STATE.json records the same curated state. This public summary does not provide all raw artifacts required to independently reproduce every historical run.

Preservation does not repair an invalid experiment or fill missing original evidence. Replay, retiming, fresh confirmation, mechanical checks and unexecuted proposals remain distinct. All stated limitations accompany the historical conclusions; no new experiment or literature review was performed for this snapshot.

#### 15. Current status

Product: **NeuroTabular 1.0.2**. Research: **frozen pending justified reopening**. Latest authority: **R38**. Primary and backup: **NONE**. Held-outs: **CLOSED**.
