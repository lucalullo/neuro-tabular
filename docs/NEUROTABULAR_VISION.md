# NeuroTabular: permanent vision and research principles

This is the single strategic reference for what NeuroTabular should become.
The [README](../README.md) describes the available product;
[AGENTS.md](../AGENTS.md) defines the working instructions for coding agents.
Roadmaps, checkpoints and experiments implement this vision without replacing it.

## Mission

Develop a significant innovation in machine learning: an original neural
algorithm for tabular data that learns accurately, generalizes reliably, trains
and predicts quickly, and uses memory efficiently.

The ambition is world-class neural tabular learning. This is a research objective,
not a claim that the current product has achieved state-of-the-art performance.
Any claim of superiority must identify its benchmark, comparators, resource
regime, uncertainty and independent verification.

## Neural identity and useful originality

The learned predictive engine must remain genuinely neural. Deterministic
preprocessing, validation and engineering support that engine. Tree models may
serve as benchmark competitors; they must not become a hidden prediction engine.

NeuroTabular should develop its own useful algorithmic identity rather than
becoming a wrapper or a renamed reproduction of another model. TabM, TabPFN,
RealMLP and other systems are references for comparison and scientific learning.
Standard components and published ideas may inform research, with proper rights
and attribution. Combining them establishes a useful contribution only when the
mechanism, distinction from prior work and benefit are demonstrated.

Originality is not complexity, a different name, a larger network or an extra
component. Seek a precise causal hypothesis, a falsifiable mechanism and evidence
that explains which conditions make it useful. Do not revive a rejected recipe
through opportunistic changes without materially new evidence.

## Objectives to improve together

| Objective | Evidence that matters |
| --- | --- |
| Accuracy | Appropriate predictive loss and task metrics on qualified real data. |
| Generalization | Independent units and dataset families, protected confirmation, robustness and stability across seeds. |
| Speed | Complete training cost, first and repeated inference, latency and throughput. |
| Memory | Temporary allocations, process RSS, VRAM where available, storage and model size, reported separately. |
| Reliability | Reproducible behavior, sound validation, persistence, compatibility and safe refitting. |
| Simplicity | An accessible API, predictable installation, clear errors and professional documentation. |

Study the Pareto frontier rather than optimizing one favorable percentage.
Prefer mechanisms that improve several dimensions. Trade-offs can be legitimate,
but their cost and intended workload must be explicit. Do not systematically
trade away predictive quality for speed, or efficiency for marginal quality.

## Failure modes that guide research

Research should address demonstrated limitations or test clearly identified
knowledge gaps, including:

- Generalization with small samples, noise, distribution shifts and limited
  statistical support.
- Rare, unknown and missing categories, including preservation of rare signals
  that are strongly predictive.
- Irrelevant features, spurious correlations, sparse interactions and complex
  nonlinear relationships.
- Numerical representation, highly sparse features, extreme or out-of-support
  inputs, and the reliability of predicted probabilities.
- Convergence, regularization, training diversity, stopping and checkpoint
  selection, distinguished from representational capacity.
- High cardinality, feature count and dataset scale, with controlled compute
  and memory requirements.

These are research questions, not a declaration that every supported workload
has every listed defect. Lower loss, smaller logits, lower gradient variance,
mathematical correctness or fewer parameters alone do not establish better
generalization or a better complete system.

## Two complementary performance frontiers

**Efficient** seeks strong quality and generalization while keeping complete
training, inference and memory costs low.

**Flagship** seeks higher predictive capability and generalization, accepting
additional compute or memory when a reproducible benefit justifies it.

They share the long-term goal of bringing accuracy and efficiency closer
together. Give each experiment its track, absolute budget, comparison regime,
success criteria and stop rules before observing results. Do not apply an
Efficient microbenchmark veto automatically to Flagship, or reclassify a failed
experiment after the fact. Scientific diagnostics can inform both frontiers
without being promoted as product improvements.

## Principles for credible evidence

1. Reuse the Scientific Atlas and original evidence before proposing an
   experiment. Distinguish measured results, hypotheses and interpretations.
2. Establish source identity, aliases, license, lineage, statistical units,
   necessary groups, duplicates, feature availability and leakage controls
   before acquiring or using data. Dataset names alone are insufficient.
3. Preserve historical reservations and frozen protocols. Separate development,
   selection and independent confirmation; fit preprocessing on training only.
   A new public copy does not remove a source family's reserved role.
4. Freeze the mathematical hypothesis, implementation, data roles, split hashes,
   seeds, metric definitions, comparisons, budgets and acceptance/regression/stop
   rules before the corresponding results. Complete formal admission before fits,
   including diagnostic training. This vision grants no admission by itself.
5. Qualify the actual integrated runner: data conversion, training, RNG behavior,
   scheduler, stopping, checkpoint restoration, persistence, interruption and
   recovery. Operator tests do not replace this qualification.
6. Use paired comparisons, multiple seeds when needed, causal controls and
   ablations. Separate architecture, capacity, optimization, preprocessing,
   tuning, refitting and ensemble effects. State when equal epochs do not mean
   equal updates or compute.
7. Count the whole pipeline, including preprocessing, validation, checkpoint I/O,
   tuning, teacher training, loading or pretraining costs where applicable.
   Document hardware, threads, runtime, warm-up, timing validity and uncertainty.
   Establish measurement validity before drawing performance conclusions.
8. Report native task metrics, uncertainty, per-dataset regressions and failures.
   Seeds and folds are not independent dataset families. Do not use incompatible
   protocols or environments as a validated global ranking.
9. Retain negative and inconclusive results, resource stops and technical
   interruptions. Do not tune thresholds, select seeds or remove unfavorable
   datasets after seeing outcomes to rescue a candidate.
10. Keep each study bounded. Continue when a distinct experiment can add useful
    knowledge; stop uninformative repetition. Research progress is measured in
    discoveries and demonstrated improvements, not document or experiment counts.

Study-specific numerical thresholds and budgets remain in their frozen
governance and protocols. This charter neither edits those records nor reopens
historically frozen research, including D37/R38.

## Scientific Atlas and permanent knowledge

The Scientific Atlas connects source and dataset characteristics to failure
modes, hypotheses, neural mechanisms, experiments, predictive quality, costs and
decisions. Every conclusion must lead back to the original evidence.

Preserve protocols, executable research code, versions, configurations, seeds,
permitted raw data, split identities, learning curves, measurements, anomalies,
decisions, checksums and recoverable checkpoints. Use the established journal,
archive and canonical indices; avoid parallel archives or duplicate mission
documents. Keep private evidence outside public documentation and distributions.

Checkpoints describe temporary state. The Atlas records scientific knowledge.
Roadmaps set changing priorities. This document supplies the enduring mission.

## From research to a stable product

Keep the Product Track and Research Track separate. Research may question the
current architecture; an unvalidated prototype must not alter the stable product.
Promotion requires:

- A reproducible, sufficiently important benefit over appropriate frozen
  baselines, confirmed independently as required by the applicable protocol.
- Qualified datasets and controls, with uncertainty, regressions and complete
  quality/compute/memory trade-offs assessed against predetermined criteria.
- Validation of the actual candidate's public API, defaults, numerical contracts,
  pandas behavior, scikit-learn integration, refit safety and persistence.
- Measured platform and dependency compatibility, package integrity, and a
  maintainable implementation without unexpected operational requirements.
- An explicit promotion decision and separate authorization for release actions.

Conservative engineering improvements can advance the Product Track with exact
valid-input equivalence and reliable complete-workload benefits. They are not
evidence of a new predictive algorithm or general accuracy improvement.

Published versions, tags, source identities and artifacts remain immutable
baselines. Future candidates must be separately identified and validated.
Technical success does not authorize a commit, push, publication or service use.
The [canonical product policies](policies/README.md) govern compatibility,
deprecation, persistence, warnings, versioning and support.

## Continuity of direction

Operational priorities and architectural hypotheses may change with evidence.
The mission remains an original neural tabular algorithm that learns better,
faster and with less memory, while remaining reliable and simple to use.

Do not silently replace that mission with the latest experiment. Surface a
material conflict and record any explicit maintainer-approved strategic revision,
preserving previous decisions. Future roadmaps should link here instead of
reproducing or redefining the full mission.
