# Results are in preparation

No genuine competitive fit, TEST prediction or ranking has been produced by
this preparation. [status.json](status.json) records this explicitly.

The executable runner and reporter will produce:

- `results.json`: per-family/seed/track/method metrics, uncertainty, costs,
  model/prediction/split hashes, A/A validity and failures.
- `results.csv`: flat measured records and failure states.
- `summary.json`: descriptive, family-weighted paired summaries with coverage.
- `REPORT.md`: comparative table with costs and limitations.
- PNG figures: quality versus complete training CPU, peak RSS and warm
  prediction latency, with separate task/track/dataset panels.

The schema is [schema.json](schema.json). The reporting implementation is
[report.py](../report.py). Test fixtures are explicitly synthetic and are not
published as benchmark evidence. Real results will be added only after the
scientific admission, execution, artifact checks and maintainer review.
