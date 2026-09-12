# Repository reorganization report

Branch: `repro/paper-v4-main-evidence`

This report records what changed, why it changed, and the repository evidence used for each step. The branch is intentionally not merged and no pull request is created.

## 1. Primary evidence package

Changed: added `reproducibility/paper_v4/` with exact derived summaries, reproduction records, protocols, source data for the transition-recovery figure, discrepancy audits, executable tests, a synthetic quick check, and a SHA-256 manifest.

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

## 3. Legacy migration and dependency resolution

Changed: moved 34 tracked files into `legacy/budget_policy_wait_abstain/`, extracted two legacy-only code/test fragments, and excluded `legacy/` from default pytest discovery. The source files are Git renames, not deletions.

Why: separate the budget, policy, and wait/abstain line from the current evidence chain. An active-module import of a legacy policy is coupling, so it was removed instead of being used as a reason to retain the policy package.

The 34 `moved_legacy` records in `MIGRATION_MANIFEST.csv` are:

```csv
configs/dummy.yaml,moved_legacy,configuration_for_legacy_baseline
configs/thumos14.yaml,moved_legacy,configuration_for_legacy_baseline
configs/thumos14_debug.yaml,moved_legacy,configuration_for_legacy_baseline
scripts/audit_v111_per_frame_schema.py,moved_legacy,legacy_schema_audit
scripts/bootstrap_ek100_transition_misalignment.py,moved_legacy,legacy_transition_analysis
scripts/diagnose_transition_sparse.py,moved_legacy,legacy_transition_analysis
scripts/evaluate.py,moved_legacy,legacy_budget_policy_runner
scripts/plot_budget_curve.py,moved_legacy,legacy_budget_plot
scripts/plot_selective_frontier.py,moved_legacy,legacy_selective_plot
scripts/plot_summary_metric.py,moved_legacy,legacy_summary_plot
scripts/plot_threshold_curve.py,moved_legacy,legacy_policy_plot
scripts/plot_transition_delay.py,moved_legacy,legacy_transition_plot
scripts/prepare_dummy_data.py,moved_legacy,legacy_baseline_helper
scripts/run_baseline.py,moved_legacy,legacy_budget_policy_runner
src/oad_stress_test/plots/__init__.py,moved_legacy,legacy_plot_package
src/oad_stress_test/plots/curves.py,moved_legacy,legacy_plot_package
src/oad_stress_test/metrics/budget.py,moved_legacy,metric_for_legacy_budget_curve
src/oad_stress_test/policies/__init__.py,moved_legacy,legacy_policy_package
src/oad_stress_test/policies/base.py,moved_legacy,legacy_policy_contract_and_state
src/oad_stress_test/policies/confidence.py,moved_legacy,legacy_wait_abstain_policies
src/oad_stress_test/policies/sampling.py,moved_legacy,legacy_budget_sampling_policies
tests/test_bootstrap_transition_misalignment.py,moved_legacy,test_for_legacy_transition_analysis
tests/test_confidence_threshold_policy.py,moved_legacy,test_for_legacy_wait_abstain_policies
tests/test_evaluate.py,moved_legacy,test_for_legacy_runner
tests/test_per_frame_logging.py,moved_legacy,test_for_legacy_logging_contract
tests/test_reliability_ladder.py,moved_legacy,test_for_legacy_reliability_ladder
tests/test_run_baseline_threshold_sweep.py,moved_legacy,test_for_legacy_runner
tests/test_shift_plumbing.py,moved_legacy,test_for_legacy_shift_path
tests/test_transition_misalignment_analyzer.py,moved_legacy,test_for_legacy_transition_analysis
tools/analyze_transition_misalignment.py,moved_legacy,legacy_transition_analysis
tools/bootstrap_transition_misalignment.py,moved_legacy,legacy_transition_analysis
tools/collect_v11_go_nogo_summary.py,moved_legacy,legacy_summary_collector
tools/reliability_ladder.py,moved_legacy,legacy_reliability_ladder
tools/validate_per_frame_log.py,moved_legacy,legacy_log_validator
```

The initial migration had 28 moved records. This review moved six additional files that had been retained only because of coupling: four files in `src/oad_stress_test/policies/`, `src/oad_stress_test/metrics/budget.py`, and `tests/test_confidence_threshold_policy.py`.

Two fragments are recorded as `extracted_legacy`:

- `make_policy()` was copied from the active factory to `legacy/budget_policy_wait_abstain/src/oad_stress_test/utils/policy_factory.py` with all constructor branches and defaults unchanged.
- `test_budget_degradation_slope_and_auc_on_toy_summary` was moved out of the active aggregate metric test into `legacy/budget_policy_wait_abstain/tests/test_metrics_budget.py`.

There are 16 `retained_dependency` records. Their actual main-tree consumers or status are:

1. `scripts/run_v112_ek100_stronger_predictor.py`: active instance pipeline entry point.
2. `scripts/run_v19_ek100_smoke.py`: active instance smoke pipeline entry point.
3. `src/oad_stress_test/evaluators/streaming.py`: imported by active evaluator, causality, GRU, TCN, linear-probe, and split tests.
4. `src/oad_stress_test/metrics/basic.py`: imported by `metrics/summary.py` and active metric tests.
5. `src/oad_stress_test/metrics/failure.py`: imported by `metrics/summary.py` and active metric tests.
6. `src/oad_stress_test/metrics/selective.py`: implementation and tests exist; the primary evidence chain does not depend on it.
7. `src/oad_stress_test/utils/factory.py`: active dataset and classifier construction used by model and split tests; the policy factory is no longer present.
8. `src/oad_stress_test/utils/jsonl_logger.py`: active execution/logging utility.
9. `tests/test_causal_evaluator.py`: active evaluator contract.
10. `tests/test_causal_gru.py`: active primary predictor tests.
11. `tests/test_causal_tcn.py`: implementation test retained; primary evidence does not depend on this predictor.
12. `tests/test_causality.py`: active no-future-input contract.
13. `tests/test_check_thumos_ready.py`: active data-readiness contract.
14. `tests/test_linear_probe.py`: active predictor, cache, and score-alignment tests.
15. `tests/test_metrics.py`: active core metrics after extraction of the budget-only test.
16. `tests/test_split_semantics.py`: active train/evaluation split contract.

Manifest verification reports `34 moved_legacy`, `2 extracted_legacy`, `16 retained_dependency`, and zero missing declared paths. The complete legacy test directory passes 55 tests when its archived source roots are explicitly enabled.

The Causal TCN implementation and selective-metric implementation remain in the main code tree and retain their tests. `audits/implementation_scope.json` records their real paths and marks them as implemented and tested but not dependencies of the primary evidence chain.

## 4. Production-code equivalence after decoupling

Two production files changed in this review.

### `streaming.py`

Before: the evaluator imported the nominal `StreamingPolicy` base class from the policy package and used it only as a type annotation. Runtime dispatch was already duck-typed.

After: the evaluator declares structural `StreamingPolicy` and `StreamingDecision` protocols locally. The bodies of `_current_feature`, `evaluate_video`, and `evaluate_dataset` are unchanged.

Evidence: AST comparison against pre-review commit `ef304b9d403f32946e1212b3da1be819c72b084c` reports `True` for each of the three runtime method bodies. Existing evaluator and causality tests still pass, and the new architecture test rejects any attempted import of `oad_stress_test.policies` while successfully importing the evaluator and active factory.

### `factory.py`

Before: the active factory imported five policy classes and exported `make_policy()` in addition to dataset and classifier constructors.

After: policy imports and `make_policy()` are removed from the active module. Dataset construction, classifier construction, cache keys, cache IO, and all model branches are unchanged. The old `make_policy()` constructor body is preserved in the legacy policy factory.

Evidence: AST comparison reports that the pre-review and extracted `make_policy()` constructor bodies are equal after ignoring the new explanatory docstring. The pre-review behavior set had 38 relevant tests; the post-review active plus extracted-legacy behavior set also has 38 passing tests. The full legacy suite, including all eight confidence-policy factory tests, passes 55 tests. The intentional API-boundary change is that active code can no longer import a legacy policy factory from `oad_stress_test.utils.factory`.

## 5. Transition-test review and coverage comparison

The three reviewed legacy files test related but different analysis lines:

- `test_bootstrap_transition_misalignment.py` has 7 tests for near/far transition error gap, abstention gap, confidence gap, per-video clustering, deterministic bootstrap backends, continuity correction, missing strata, and output schemas.
- `test_transition_misalignment_analyzer.py` has 10 tests for nearest-transition distance, distance bins, raw/selective error columns, multilabel-column preservation, odds ratios, output/figure/regression modes, and exclusion of videos without transitions.
- `test_reliability_ladder.py` has 14 tests for causal/oracle regimes, active-label-set transition events, global and Mondrian thresholds, cluster bootstrap, archive and official-split validation, raw-error semantics, and adaptive coverage procedures.

None directly executes the primary `transition_stats()` contract for clean adjacent labels, the finite `[0, 16)` recovery window, next-transition truncation, and miss-as-16 scoring. Moving any whole file back would therefore mix legacy selective-reliability analysis into the primary test surface. All three remain in legacy; their 31 tests pass in place.

The original four `paper_v4` metric-protocol tests covered active-set membership, rejection when the pre-transition side is multi-active, a combined transition/miss example, and the frozen protocol fields. Four direct gaps were added:

1. Post-transition clean filtering: rejects a transition when the first post-transition timestep is multi-active.
2. Adversarial next-transition truncation: places the old target class only after the next true boundary; looking beyond that boundary incorrectly changes a miss at 16 into a hit at delay 2.
3. Exact half-open horizon: offset 15 is accepted, while offset 16 is excluded and scored as a miss with delay 16.
4. Degenerate sequence: a sequence with no ground-truth transitions returns zero count, zero delay sum, and zero miss sum.

Each added test was mutation-checked independently:

- Removing the post-transition active-count condition made test 1 fail (`transition_count` became 1 instead of 0).
- Removing `next_boundary` from the window end made test 2 fail (`delay_sum` became 2 and `missed_sum` became 0 instead of 16 and 1).
- Extending the end to `boundary + horizon + 1` made test 3 fail because the offset-16 case was accepted rather than missed.
- Returning `eligible + 1` made test 4 fail because a flat sequence reported one transition.

All temporary mutations were reverted. The final implementation passes all eight direct metric-protocol tests.

## 6. Quick check

Changed: replaced the two-transition toy sequence with a deterministic 21-segment, 20-transition sequence.

Why: avoid a denominator so small that one event produces an exaggerated 50-percentage-point miss-rate change.

Final native-to-EMA deltas are:

- accuracy: `+0.004987531172069848`
- expected calibration error: `-0.07049772133210579`
- mean transition delay: `+0.8` timestep
- missed-transition rate: `+0.05` (5 percentage points)

These are close in scale to the frozen primary instance (`+0.00575709695922527`, `-0.06112126224932643`, `+0.8851607382960527`, and `+0.0558611416697225`). The test freezes the 20-transition count and all four synthetic deltas.

## 7. Discrepancies and frozen literals

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

## 8. Source data copy

Changed: copied the existing 64-row source CSV and its audit JSON into `reproducibility/paper_v4/figure3_source/`, then marked both files `-text` in `.gitattributes` so Git preserves their original CRLF bytes.

Why: put the figure source in version control while maintaining byte-for-byte auditability.

Evidence:

- CSV source SHA-256: `0fdcd9ef676b2cfefad0d37116c9087a866160350e717e28b7a09b7f99b6ecd7`
- CSV repository SHA-256: `0fdcd9ef676b2cfefad0d37116c9087a866160350e717e28b7a09b7f99b6ecd7`
- Result: exact match
- Audit JSON source and repository SHA-256: `366464c2eca2d12bdb8c28c37d3c97dc0f9c16b5d0ca029b1002f3e6a0adc380`

## 9. Test accounting

Baseline: 148 tests passed before repository reorganization.

Tests moved out of default discovery: 55 total.

- The original seven archived legacy files contain 46 tests.
- `test_confidence_threshold_policy.py` adds 8 archived tests.
- The extracted budget-metric test adds 1 archived test.

Tests added to default discovery: 19 total.

- `reproducibility/paper_v4/tests`: 18 tests (4 quick-check, 8 metric-protocol, 2 five-instance boxcar CI, and 4 package-integrity tests).
- `tests/test_main_tree_boundaries.py`: 1 architecture-boundary test.

Net change: `+19 - 55 = -36`. Final active suite: `112 passed`, exactly `148 - 36`.

Package-specific verification:

- `paper_v1`: 10 passed
- `paper_v2`: 4 passed
- `paper_v3`: 5 passed
- `paper_v4`: 18 passed
- complete active suite: 112 passed
- complete explicit legacy suite: 55 passed

## 10. Metadata and integrity checks

Changed: prepared a neutral repository description and topics for the remote repository. The authenticated GitHub plugin verifies repository access and the uploaded branch, but the installed connector has no repository-description or topic mutation operation; those two remote-homepage fields remain pending. No author metadata in `CITATION.cff` is changed.

Evidence:

- `CITATION.cff` parses as CFF 1.2 YAML and has no diff from `main`.
- Continuous integration invokes `python -m pytest -q`.
- No excluded checkpoint, prediction-cache, or bootstrap-array file is tracked.
- The package checksum manifest contains 20 entries and verifies with zero mismatches.
- `git diff --check` reports no whitespace errors.
