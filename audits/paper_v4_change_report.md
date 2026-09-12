# Repository reorganization report

Branch: `repro/paper-v4-main-evidence`

This report records what changed, why it changed, and the repository evidence used for each step. The branch is intentionally not merged and no pull request is created.

## 1. Primary evidence package

Changed: added `reproducibility/paper_v4/` with exact derived summaries, reproduction records, protocols, Figure 3 source data, discrepancy audits, executable tests, a synthetic quick check, and a SHA-256 manifest.

Why: bind the repository to one manuscript version without rewriting the published `paper_v2` and `paper_v3` packages.

Evidence:

- EMA point estimates and bootstrap deltas are filtered copies of recovered frozen summaries; source and output hashes are recorded in `derived_summaries/ek100_causal_gru_ema/provenance.json`.
- TeSTra and CMeRT reproductions are recorded under `reproduction_logs/` with source paths, SHA-256 digests, and sample counts.
- Full instance membership, timestep counts, transition counts, training settings, metric rules, and bootstrap settings are recorded under `protocols/`.
- `git diff main -- reproducibility/paper_v2 reproducibility/paper_v3` is empty.

Large artifacts are not published. `trained_model.pt`, `frozen_predictions.npz`, and `bootstrap_samples.npz` remain outside the repository; their sizes and SHA-256 digests are preserved in provenance.

## 2. Extended analyses

Changed: added `reproducibility/extended/README.md` as an index of temporal flip and switch counts, coefficient 0.25, multi-bin calibration scans, alternate horizons, leave-one-video-out checks, and all-transition analyses.

Why: make their non-primary status explicit without deleting, moving, or changing their existing data files.

Evidence: every index entry points to a file that already exists in a historical reproducibility package.

## 3. Legacy baseline

Changed: moved 28 tracked files into `legacy/budget_policy_wait_abstain/`, added a migration manifest, and excluded `legacy/` from default pytest discovery. The moves are recorded as Git renames, not deletions.

Why: separate an earlier budget, policy, and wait/abstain line from the current evidence chain while preserving its code, tests, configurations, and utilities.

Evidence: `legacy/budget_policy_wait_abstain/MIGRATION_MANIFEST.csv` contains 28 `moved_legacy` records and 21 `retained_dependency` records. All 49 recorded paths exist at their declared locations.

Files retained in the main tree under the dependency rule:

1. `scripts/run_v112_ek100_stronger_predictor.py`
2. `scripts/run_v19_ek100_smoke.py`
3. `src/oad_stress_test/evaluators/streaming.py`
4. `src/oad_stress_test/metrics/basic.py`
5. `src/oad_stress_test/metrics/budget.py`
6. `src/oad_stress_test/metrics/failure.py`
7. `src/oad_stress_test/metrics/selective.py`
8. `src/oad_stress_test/policies/base.py`
9. `src/oad_stress_test/policies/confidence.py`
10. `src/oad_stress_test/policies/sampling.py`
11. `src/oad_stress_test/utils/factory.py`
12. `src/oad_stress_test/utils/jsonl_logger.py`
13. `tests/test_causal_evaluator.py`
14. `tests/test_causal_gru.py`
15. `tests/test_causal_tcn.py`
16. `tests/test_causality.py`
17. `tests/test_check_thumos_ready.py`
18. `tests/test_confidence_threshold_policy.py`
19. `tests/test_linear_probe.py`
20. `tests/test_metrics.py`
21. `tests/test_split_semantics.py`

The Causal TCN implementation and selective-metric implementation remain in the main code tree and retain their tests. `audits/implementation_scope.json` marks them as implemented and tested but not dependencies of the primary evidence chain.

## 4. README and minimal verification

Changed: rewrote the root README with a neutral opening, actual repository contents, a synthetic quick check, runnable verification paths, relative directory links, an extended-analysis section, a legacy notice, artifact policy, and known discrepancies.

Why: ensure the public entry point reflects the current tree and the primary evidence package without exposing review-sensitive identity information.

Evidence: `python reproducibility/paper_v4/quick_check.py` completes without writing files and reports native versus EMA accuracy, expected calibration error, mean transition delay, and missed-transition rate. Its three tests pass.

## 5. Discrepancies and frozen literals

Changed: added a four-field mismatch audit for the CMeRT boxcar accuracy delta and preserved an additional suspected-origin field.

Why: retain disagreement as an audit asset without modifying either the frozen value or the displayed value.

Evidence:

- `source_exact`: `-0.007333105779500304`
- `correct_4dp`: `-0.0073`
- `manuscript_display`: `-0.0077`
- `audit_status`: `mismatch_preserved_unresolved`
- `suspected_origin`: `-0.007723984541020301`
- `suspected_origin_status`: `unverified_hypothesis_not_an_explanation`

The three generation literals have frozen sources in `reproducibility/paper_v3/derived_summaries/cmert_thumos/bootstrap_deltas.csv`:

- `-0.0018` from exact `-0.0017828539082419478`
- `+0.9838` from exact `0.9838187702265371`
- `+0.0331` from exact `0.033123929183323825`

Their mapping and source digest are recorded in `audits/manuscript_literal_trace.json`.

## 6. Figure 3 source copy

Changed: copied the existing 64-row CSV and its audit JSON into `reproducibility/paper_v4/figure3_source/`, then marked both files `-text` in `.gitattributes` so Git preserves their original CRLF bytes.

Why: move the source data into version control while maintaining byte-for-byte auditability.

Evidence:

- CSV source SHA-256: `0fdcd9ef676b2cfefad0d37116c9087a866160350e717e28b7a09b7f99b6ecd7`
- CSV repository SHA-256: `0fdcd9ef676b2cfefad0d37116c9087a866160350e717e28b7a09b7f99b6ecd7`
- Result: exact match
- Audit JSON source and repository SHA-256: `366464c2eca2d12bdb8c28c37d3c97dc0f9c16b5d0ca029b1002f3e6a0adc380`

## 7. Test accounting

Baseline: 148 tests passed before this branch.

Changes:

- Added 13 primary-package tests: 3 quick-check tests, 4 metric-protocol tests, 2 five-instance boxcar CI tests, and 4 package-integrity tests.
- Archived 7 legacy test files containing 46 collected tests and excluded `legacy/` from default discovery.
- Net change: `+13 - 46 = -33` tests.
- Final active suite: `115 passed`, exactly `148 - 33`.

## 8. Metadata and integrity checks

Changed: prepared a neutral repository description and topics for the remote repository. No author metadata in `CITATION.cff` is changed.

Evidence:

- `CITATION.cff` parses as CFF 1.2 YAML and has no diff from `main`.
- Continuous integration invokes `python -m pytest -q`.
- No excluded checkpoint, prediction-cache, or bootstrap-array file is tracked.
- The package checksum manifest contains 20 entries and verifies with zero mismatches.
- `git diff --check` reports no whitespace errors.
