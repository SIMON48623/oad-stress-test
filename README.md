# OAD stress-test repository

This repository provides causal online action-detection evaluation code, frozen derived summaries, explicit protocols, and audit records for studying the reliability effects of temporal postprocessing. The primary evidence package is versioned separately from earlier releases and exploratory analyses.

## Contents

- Causal data adapters and tested predictor implementations for linear and recurrent models.
- Accuracy, calibration, transition-delay, and missed-transition metrics.
- Frozen point estimates, clustered-bootstrap deltas, provenance records, and reproduction logs.
- Protocol tests for active-set top-1 scoring, clean-transition selection, finite transition windows, and missed-transition handling.
- A byte-preserved source table and audit record for the transition-recovery figure.
- Earlier published reproducibility packages, extended analyses, and an archived legacy baseline.

The implementation-scope audit records additional tested modules that exist in the codebase but are not dependencies of the primary evidence chain: [`audits/implementation_scope.json`](audits/implementation_scope.json).

## Installation

Python 3.9 or later is required.

```bash
git clone <repository-url>
cd oad-stress-test
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

PyTorch is optional for the core evaluation and synthetic check. It is required only for model training and tests that exercise torch-backed predictors.

## Quick check

This deterministic synthetic comparison completes in a few seconds, needs no research dataset, and writes no result files:

```bash
python reproducibility/paper_v4/quick_check.py
```

It prints native and EMA values for accuracy, expected calibration error, mean transition delay, and missed-transition rate. The expected output is:

```text
condition  accuracy      ece           mean_transition_delay  missed_transition_rate
native     0.8333333333  0.2583333333  0.0000000000           0.0000000000
ema        0.8888888889  0.1712639279  8.0000000000           0.5000000000
```

## Primary reproducibility package

The single-version evidence package is in [`reproducibility/paper_v4/`](reproducibility/paper_v4/). Its checks require only the repository's development dependencies:

```bash
python -m pytest -q reproducibility/paper_v4/tests
```

The package includes derived summaries, reproduction logs, split and metric protocols, source data for the transition-recovery figure, discrepancy audits, tests, and a complete checksum manifest. Large checkpoints, frozen probability arrays, and bootstrap draw arrays are intentionally excluded; their source paths, sizes, and SHA-256 digests are recorded in provenance instead.

Earlier versioned packages remain unchanged in [`reproducibility/paper_v1/`](reproducibility/paper_v1/), [`reproducibility/paper_v2/`](reproducibility/paper_v2/), and [`reproducibility/paper_v3/`](reproducibility/paper_v3/). They are retained for historical traceability and are not rewritten by the current package.

## Extended analyses

Temporal flip counts, prediction-switch counts, coefficient 0.25, multi-bin calibration scans, alternate transition horizons, leave-one-video-out checks, and all-transition analyses are indexed in [`reproducibility/extended/`](reproducibility/extended/). They remain available but are outside the primary evidence chain.

## Known discrepancies

Known numerical discrepancies are preserved as audit assets rather than changed to match display text:

- [`reproducibility/paper_v4/audits/cmert_boxcar_accuracy_mismatch.json`](reproducibility/paper_v4/audits/cmert_boxcar_accuracy_mismatch.json) records the frozen accuracy delta, its correct four-decimal rendering, the different displayed value, and an unresolved suspected origin that is explicitly not treated as an explanation.
- [`audits/manuscript_literal_trace.json`](audits/manuscript_literal_trace.json) records the frozen sources for three literals used by the manuscript-generation path.
- [`reproducibility/paper_v4/derived_summaries/ek100_causal_gru_ema/provenance.json`](reproducibility/paper_v4/derived_summaries/ek100_causal_gru_ema/provenance.json) localizes the recovered run-log checksum mismatch and preserves both digests.

No value is edited to remove a discrepancy. See [`reproducibility/paper_v4/audits/`](reproducibility/paper_v4/audits/) for the five-instance confidence-interval audit.

## Repository layout

```text
audits/                 repository-level scope and source-trace audits
docs/                   repository design and implementation records
legacy/                 preserved baseline from a separate research line
reproducibility/        versioned evidence packages and extended-analysis index
scripts/                active data preparation and replay entry points
src/oad_stress_test/    installable Python package
tests/                  active unit and pipeline tests
tools/                  active validation utilities
```

The archived budget, policy, and wait/abstain baseline is in [`legacy/budget_policy_wait_abstain/`](legacy/budget_policy_wait_abstain/). It is retained intact for traceability but does not participate in primary conclusions.

## Research data and artifact policy

Research datasets, third-party features, checkpoints, frozen prediction caches, and bootstrap draw arrays are not redistributed. Obtain datasets from their original providers and follow [`DATASET.md`](DATASET.md) for local layout guidance. The repository publishes derived summaries and cryptographic provenance for excluded large artifacts.

The ignore rules exclude common data and model formats from accidental commits. The versioned CSV and JSON evidence records are intentionally trackable.

## Testing and continuous integration

Run the active suite with:

```bash
python -m pytest -q
```

The continuous-integration workflow executes this pytest command on supported Python versions. Archived legacy tests are preserved under `legacy/` and excluded from default discovery.

## Citation and license

Use the unchanged author metadata in [`CITATION.cff`](CITATION.cff). Original source code is released under the [MIT License](LICENSE). Third-party datasets, annotations, features, and model artifacts are not covered by this license.
