# Third-party notices

NeuroTabular calls public APIs from the runtime dependencies below. Package
managers install them separately; their source and binaries are not copied into
the NeuroTabular wheel.

| Dependency | Upstream license | Purpose |
| --- | --- | --- |
| [NumPy](https://github.com/numpy/numpy/blob/main/LICENSE.txt) | BSD-3-Clause, with distribution-specific bundled notices | Array operations |
| [pandas](https://github.com/pandas-dev/pandas/blob/main/LICENSE) | BSD-3-Clause, with distribution-specific bundled notices | DataFrame input and dtype handling |
| [PyTorch](https://github.com/pytorch/pytorch/blob/main/LICENSE) | BSD-style primary license plus notices for bundled components | Neural network and optimization engine |
| [scikit-learn](https://github.com/scikit-learn/scikit-learn/blob/main/COPYING) | BSD-3-Clause | Estimator conventions, splitting, and metrics |

Development tools such as pytest, Ruff, build, Twine, and psutil are not runtime
dependencies. Benchmark-only comparisons may use separately installed external
ML libraries; they are optional, are not imported by NeuroTabular training or
inference, and are governed by their own licenses.

Dependency wheels can bundle numerical libraries, compiler runtimes, or other
components with additional notices. Redistributors must inspect and preserve
the notices required by the exact artifacts and platforms they distribute.
This is an engineering notice, not legal advice.
