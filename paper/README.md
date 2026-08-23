# Paper artifact

`main.md` is the submission-shaped internal manuscript. It is intentionally a
technical-report/workshop draft rather than an ICRA/RSS claim: the executed
effect head did not improve routing and the frozen executor failed the
preregistered full-loop gate.

The manuscript is evidence-linked to immutable JSON, raw public frames/action
histories, task-specific evaluator relabels, and grouped development CV. Before
external submission, replace the anonymous placeholder, select a venue
template, expand the seed/group and primitive coverage, and run a new untouched
calibration/test split. Do not rewrite the current negative result.

The successor's machine-generated table snapshot is
[`generated/piu_evidence_tables_v1.md`](generated/piu_evidence_tables_v1.md).
It is deliberately incomplete: every unavailable real training, calibration,
or sealed artifact is `PENDING`, not zero. The corresponding JSON is generated
from hash-checked sources by `scripts/evaluation/build_piu_paper_tables.py` and
the v1 snapshot is immutable.

The method and evidence-boundary figures are likewise deterministic SVGs:
[`generated/piu_method_pipeline_v1.svg`](generated/piu_method_pipeline_v1.svg)
and
[`generated/piu_evidence_boundary_v1.svg`](generated/piu_evidence_boundary_v1.svg).
They are generated from the frozen figure protocol and hash-checked evidence
table by `scripts/evaluation/build_piu_paper_figures.py`; `--verify` compares
both files byte-for-byte.
