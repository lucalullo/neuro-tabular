# Legal, provenance, and naming notes

This document records preliminary engineering due diligence. It is not legal
advice, a freedom-to-operate opinion, or professional trademark clearance. The
name **NeuroTabular** is not represented or guaranteed to be legally available.

## Code and data provenance

- NeuroTabular source, tests, documentation, examples, and benchmark harnesses
  were written for this project.
- The implementation calls public APIs from NumPy, pandas, PyTorch, and
  scikit-learn. It does not vendor or modify their source.
- Public scientific papers and official performance documentation informed
  independently written design decisions recorded in `RESEARCH_NOTES.md`.
- No source implementation was copied from external tabular-learning projects.
- Tests, examples, and release benchmarks use synthetic data. No third-party
  dataset is distributed.
- Training and inference contain no telemetry, hosted-service client, upload,
  account requirement, or network request.

Contributors must submit original work or material they are authorized to
license. Users remain responsible for data governance, privacy, security,
regulatory obligations, and the consequences of models trained on their data.

## Preliminary naming review on 2026-08-27

Exact-name web, PyPI, and GitHub searches performed for this engineering review
did not identify a clearly indexed Python package or software repository using
the exact project name. Search engines and indexes can be incomplete, delayed,
or jurisdictionally narrow; the result does not establish availability.

The review identified the earlier EU word mark **NEUROTAB**, application number
`015727282`, associated with software and medical/neurostimulation-related
goods. **NeuroTabular contains the complete string “NeuroTab” and identifies
software. This creates a material naming risk that requires professional review
before public launch, marketing, registration, or commercial reliance.**

The exact status, renewal state, goods and services, ownership, and geographic
scope must be checked directly in official registers immediately before any
publication decision. Similarity analysis depends on jurisdiction, goods,
services, consumers, presentation, and other legal factors that this engineering
review cannot determine.

Official starting points include:

- [EUIPO search services](https://www.euipo.europa.eu/en/search-ip)
- [EUIPO eSearch](https://euipo.europa.eu/eSearch/)
- [TMview](https://www.tmdn.org/tmview/)
- [WIPO Global Brand Database](https://branddb.wipo.int/)

Searches can miss unregistered rights, recent filings, translations, similar
marks, non-indexed uses, and relevant national rights. Package, repository, or
domain availability does not create trademark rights.

## Before publication

Repeat exact and similarity searches on PyPI, GitHub, the web, EUIPO/TMview,
WIPO, and relevant national registers. Obtain professional trademark advice for
intended markets, particularly software-related classes and any medical or
neurological use. Review the exact dependency artifacts and notices, scan the
repository and history for credentials or private data, and confirm that all
contributors have supplied original authorized work.
