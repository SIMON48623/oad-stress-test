# Paper v4 Repository Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an anonymous, checksum-verified `paper_v4` evidence package, separate extended and legacy material, and provide a deterministic synthetic quick check without changing frozen results.

**Architecture:** Add a provenance-first evidence layer over byte-unchanged historical packages. Move only baseline-primary files into a path-preserving legacy snapshot, retain shared dependencies in the main tree, and validate the result through behavior tests, content hashes, anonymity scans, and Git history comparisons.

**Tech Stack:** Python 3.10+, NumPy, pytest, JSON, CSV, Git, PowerShell SHA-256 tooling, GitHub CLI.

**Spec:** `docs/superpowers/specs/2026-09-12-paper-v4-repository-design.md`

## Global Constraints

- Do not delete frozen experimental data, bootstrap output, provenance, discrepancy records, or audits.
- Do not change any numeric value to force agreement.
- Do not create experimental results; only copy frozen summaries and create documentation, tests, checksums, and the synthetic quick check.
- Keep `reproducibility/paper_v2/` and `reproducibility/paper_v3/` byte-for-byte unchanged.
- Preserve the `CITATION.cff` author value exactly.
- Do not add identifying manuscript metadata to the root README, new documentation, or commit messages.
- Exclude `trained_model.pt`, `frozen_predictions.npz`, and `bootstrap_samples.npz`; record sizes and SHA-256 only.
- Keep `src/oad_stress_test/models/causal_tcn.py` and `src/oad_stress_test/metrics/selective.py` in the main tree.
- Push `repro/paper-v4-main-evidence` only; do not merge and do not create a pull request.

---

### Task 1: Freeze source inventory and derived evidence

**Files:**
- Create: `reproducibility/paper_v4/derived_summaries/ek100_causal_gru_ema/point_estimates.csv`
- Create: `reproducibility/paper_v4/derived_summaries/ek100_causal_gru_ema/bootstrap_deltas.csv`
- Create: `reproducibility/paper_v4/derived_summaries/ek100_causal_gru_ema/provenance.json`
- Create: `reproducibility/paper_v4/reproduction_logs/testra_ek100.json`
- Create: `reproducibility/paper_v4/reproduction_logs/cmert_thumos14.json`
- Create: `reproducibility/paper_v4/protocols/instance_splits_and_counts.json`
- Create: `reproducibility/paper_v4/protocols/training_config.json`
- Create: `reproducibility/paper_v4/protocols/bootstrap_protocol.json`
- Create: `reproducibility/paper_v4/protocols/metric_protocol.json`

**Interfaces:**
- Consumes: recovered `oad_g9_results` summaries and hashes; frozen `paper_v1`, `paper_v2`, and `paper_v3` protocols and summaries.
- Produces: public, source-addressed JSON/CSV evidence consumed by package tests and README links.

- [ ] **Step 1: Record immutable source digests**

Verify these source facts before writing:

```text
recovered point_estimates.csv sha256 dcbe65d56e9d5907c8ae4cd62406c4337edc5ccb8ec9c309e022d62473d5091e
recovered bootstrap_deltas.csv sha256 1db48176e9e369cc7a786e2c027ee6265a584b71d2f016aaa13a8b956e18edf8
recovered trained_model.pt sha256 a0acf9bd2a4d6d349aad71f264b63082680f8bc6ec534aff2c22e453d7451123
recovered frozen_predictions.npz sha256 e9e2ccaf00ad689618846cee0b2b5a3241c71cc017ff6673be4935b9ce4e0ffc
recovered bootstrap_samples.npz sha256 24893bf30eebe66913f5619f0ea60e312832a047ffe19541c63056f656db2f04
```

- [ ] **Step 2: Copy only manuscript-facing recovered rows**

Use `apply_patch` to reproduce the source CSV text exactly for `point_estimates.csv`. For `bootstrap_deltas.csv`, include only `ema_0.50_minus_raw` rows for primary accuracy, global ECE, clean delay, and clean missed-transition rate; preserve every selected decimal exactly and record that this is a row-filtered copy in provenance. Do not include alpha 0.25, TFI, switch counts, all-transition rows, leave-one-out rows, or bootstrap samples in the main package.

- [ ] **Step 3: Write recovery provenance**

Record source logical paths, recovery date, source/output hashes, excluded artifact hashes and sizes, and both formal-log digests. Set the checksum incident to `preserved_known_mismatch` and its cause to the `tee`/post-digest append ordering proven by `run_formal.sh`.

- [ ] **Step 4: Write reproduction logs**

Copy exact metrics and sample counts from:

```text
reproducibility/paper_v2/derived_summaries/testra_ek100/official_reproduction.json
reproducibility/paper_v2/derived_summaries/testra_ek100/point_estimates.csv
reproducibility/paper_v3/derived_summaries/cmert_thumos/official_reproduction.json
reproducibility/paper_v3/derived_summaries/cmert_thumos/point_estimates.csv
```

The TeSTra record reports `30.76995422935509`, 138 sequences, 189943 timesteps, 10301 transitions, and 9668 validation segments. The CMeRT record reports mAP percentages `73.221` and `59.442`, 213 videos, 181642 valid outputs, and 5253 clean transitions.

- [ ] **Step 5: Write split and protocol records**

Record the THUMOS14 211-video split digest and excluded IDs `video_test_0000270` and `video_test_0001292`; record CMeRT's 213-video official split; record the recovered EK100 500/133 split, 981347 training timesteps, 284171 evaluation timesteps, 21890 primary transitions, and 16362 clean transitions. Include source-declared counts for the five manuscript instances and explicit `not_reported_in_frozen_source` values where a count is absent.

- [ ] **Step 6: Verify source equality and selected rows**

Run a PowerShell hash comparison for the full copied point-estimate file and a CSV row comparison for selected bootstrap rows. Expected: point-estimate digest equals `dcbe65...d5091e`; every public bootstrap row has an exact matching source row.

- [ ] **Step 7: Commit evidence package**

```bash
git add reproducibility/paper_v4/derived_summaries reproducibility/paper_v4/reproduction_logs reproducibility/paper_v4/protocols
git commit -m "repro: add versioned evidence records"
```

### Task 2: Add the deterministic quick check with TDD

**Files:**
- Create: `reproducibility/paper_v4/tests/test_quick_check.py`
- Create: `reproducibility/paper_v4/quick_check.py`

**Interfaces:**
- Produces: `ema_probabilities(probabilities, alpha)`, `expected_calibration_error(labels, probabilities, bins)`, `transition_summary(labels, predictions, horizon)`, `run_quick_check()`, and a JSON CLI report.
- Consumes: NumPy only; no datasets, checkpoints, caches, or output paths.

- [ ] **Step 1: Write the failing behavior tests**

Tests use literal synthetic expectations and assert that:

```python
result = quick_check.run_quick_check()
assert set(result) == {"native", "ema_alpha_0.50"}
assert set(result["native"]) == {
    "accuracy", "ece", "mean_transition_delay", "missed_transition_rate"
}
assert result["ema_alpha_0.50"]["accuracy"] > result["native"]["accuracy"]
assert result["ema_alpha_0.50"]["ece"] < result["native"]["ece"]
assert result["ema_alpha_0.50"]["mean_transition_delay"] > result["native"]["mean_transition_delay"]
assert result["ema_alpha_0.50"]["missed_transition_rate"] > result["native"]["missed_transition_rate"]
```

Also assert a two-transition fixture where one miss produces delay `16`, yielding mean delay `8.0` and miss rate `0.5`.

- [ ] **Step 2: Run RED**

Run:

```bash
python -m pytest -q reproducibility/paper_v4/tests/test_quick_check.py
```

Expected: collection/import failure because `quick_check.py` does not exist.

- [ ] **Step 3: Implement the minimum quick check**

Use a deterministic 18-timestep, two-class fixture with a two-timestep middle action, three native one-step glitches, EMA alpha `0.50`, ECE bins fixed in the script, and horizon `16`. The transition window stops at the next label transition; a miss contributes delay `16`.

- [ ] **Step 4: Run GREEN and the CLI**

Expected: tests pass; CLI prints one JSON object and creates no files.

- [ ] **Step 5: Commit quick check**

```bash
git add reproducibility/paper_v4/quick_check.py reproducibility/paper_v4/tests/test_quick_check.py
git commit -m "test: add synthetic comparison check"
```

### Task 3: Add protocol, discrepancy, Figure 3, and integrity audits

**Files:**
- Create: `reproducibility/paper_v4/tests/test_metric_protocol.py`
- Create: `reproducibility/paper_v4/tests/test_boxcar_ci.py`
- Create: `reproducibility/paper_v4/tests/test_package_integrity.py`
- Create: `reproducibility/paper_v4/audits/five_instance_boxcar_ci.csv`
- Create: `reproducibility/paper_v4/audits/cmert_boxcar_accuracy_mismatch.json`
- Create: `reproducibility/paper_v4/audits/displayed_value_trace.json`
- Create: `reproducibility/paper_v4/audits/implementation_scope.json`
- Create: `reproducibility/paper_v4/figure3_source/figure3_transition_recovery.csv`
- Create: `reproducibility/paper_v4/figure3_source/figure3_audit.json`
- Create: `reproducibility/paper_v4/figure3_source/provenance.json`
- Create: `reproducibility/extended/README.md`

**Interfaces:**
- Consumes: existing protocol functions, five frozen boxcar sources, manuscript generator audit values, and the external Figure 3 CSV/JSON.
- Produces: executable protocol assertions and source-addressed discrepancy/figure records.

- [ ] **Step 1: Write protocol tests before audit artifacts**

Exercise real existing functions:

```python
top1_in_active_set([2, 1], [[1, 2], [2]]) == [True, False]
transition_stats(labels, predictions, active_counts, horizon=16)
```

Fixtures independently assert clean-transition exclusion, next-transition truncation, horizon `16`, and miss-delay assignment `16`.

- [ ] **Step 2: Write the five-instance CI test**

Load the audit CSV and require exactly these instances:

```text
thumos14_causal_gru
thumos14_linear_probe
ek100_causal_gru
testra_ek100
cmert_thumos14
```

For every instance require ECE `ci_high < 0`, delay `ci_low > 0`, and missed-transition `ci_low > 0`. Accuracy is recorded but is not part of this five-instance direction assertion.

- [ ] **Step 3: Run RED**

Expected: tests fail because the audit CSV and package records do not yet exist.

- [ ] **Step 4: Add exact audit artifacts**

Populate the CI CSV only from existing bootstrap files. Add mismatch fields:

```json
{
  "source_exact": -0.007333105779500304,
  "correct_4dp": -0.0073,
  "manuscript_display": -0.0077,
  "audit_status": "mismatch_preserved_unresolved",
  "suspected_origin": -0.007723984541020301,
  "suspected_origin_status": "unverified_hypothesis_not_an_explanation"
}
```

Trace `-0.0018`, `+0.9838`, and `+0.0331` to their exact CMeRT boxcar rows. Record TCN and selective implementations as present but not manuscript-dependent.

- [ ] **Step 5: Copy Figure 3 exactly**

Use `apply_patch` to add the 64 data rows and audit JSON without edits. Record source digests:

```text
CSV  0fdcd9ef676b2cfefad0d37116c9087a866160350e717e28b7a09b7f99b6ecd7
JSON 366464c2eca2d12bdb8c28c37d3c97dc0f9c16b5d0ca029b1002f3e6a0adc380
```

- [ ] **Step 6: Add the extended-analysis index**

Link real retained paths for TFI, switch counts, alpha `0.25`, ECE-bin scans, `H=8/32`, leave-one-out, and all-transition analyses. State that none belongs to the main evidence chain.

- [ ] **Step 7: Run GREEN and commit**

```bash
python -m pytest -q reproducibility/paper_v4/tests/test_metric_protocol.py reproducibility/paper_v4/tests/test_boxcar_ci.py reproducibility/paper_v4/tests/test_package_integrity.py
git add reproducibility/paper_v4 reproducibility/extended
git commit -m "repro: add protocol and discrepancy audits"
```

### Task 4: Move the legacy baseline and preserve dependency exceptions

**Files:**
- Create: `legacy/budget_policy_wait_abstain/README.md`
- Create: `legacy/budget_policy_wait_abstain/MIGRATION_MANIFEST.csv`
- Move: 28 path-preserved files listed below
- Modify: `pyproject.toml`

**Interfaces:**
- Consumes: the baseline-primary classification and current import graph.
- Produces: a preserved legacy snapshot excluded from root test discovery, plus a machine-readable moved/retained audit.

- [ ] **Step 1: Move the exact approved closure**

Move these files with `git mv` into `legacy/budget_policy_wait_abstain/<original-path>`:

```text
configs/dummy.yaml
configs/thumos14.yaml
configs/thumos14_debug.yaml
scripts/audit_v111_per_frame_schema.py
scripts/bootstrap_ek100_transition_misalignment.py
scripts/diagnose_transition_sparse.py
scripts/evaluate.py
scripts/plot_budget_curve.py
scripts/plot_selective_frontier.py
scripts/plot_summary_metric.py
scripts/plot_threshold_curve.py
scripts/plot_transition_delay.py
scripts/prepare_dummy_data.py
scripts/run_baseline.py
src/oad_stress_test/plots/__init__.py
src/oad_stress_test/plots/curves.py
tests/test_bootstrap_transition_misalignment.py
tests/test_evaluate.py
tests/test_per_frame_logging.py
tests/test_reliability_ladder.py
tests/test_run_baseline_threshold_sweep.py
tests/test_shift_plumbing.py
tests/test_transition_misalignment_analyzer.py
tools/analyze_transition_misalignment.py
tools/bootstrap_transition_misalignment.py
tools/collect_v11_go_nogo_summary.py
tools/reliability_ladder.py
tools/validate_per_frame_log.py
```

- [ ] **Step 2: Record 21 dependency-retained keyword matches**

Create manifest rows for these retained files and their reasons:

```text
scripts/run_v112_ek100_stronger_predictor.py
scripts/run_v19_ek100_smoke.py
src/oad_stress_test/evaluators/streaming.py
src/oad_stress_test/metrics/basic.py
src/oad_stress_test/metrics/budget.py
src/oad_stress_test/metrics/failure.py
src/oad_stress_test/metrics/selective.py
src/oad_stress_test/policies/base.py
src/oad_stress_test/policies/confidence.py
src/oad_stress_test/policies/sampling.py
src/oad_stress_test/utils/factory.py
src/oad_stress_test/utils/jsonl_logger.py
tests/test_causal_evaluator.py
tests/test_causal_gru.py
tests/test_causal_tcn.py
tests/test_causality.py
tests/test_check_thumos_ready.py
tests/test_confidence_threshold_policy.py
tests/test_linear_probe.py
tests/test_metrics.py
tests/test_split_semantics.py
```

The retained reasons distinguish shared causal/model test dependencies from the explicit TCN/selective exception.

- [ ] **Step 3: Exclude legacy from supported test discovery**

Add `norecursedirs = ["legacy"]` under `[tool.pytest.ini_options]`. Do not delete or rename legacy tests.

- [ ] **Step 4: Run the full suite and explain count changes**

Collect tests before and after migration. Compute the exact number removed from supported discovery and the exact number newly added under `paper_v4`; reconcile the final total with the 148-test baseline.

- [ ] **Step 5: Commit legacy migration**

```bash
git add legacy pyproject.toml configs scripts src tests tools
git commit -m "chore: archive legacy baseline paths"
```

### Task 5: Rewrite the root README and package README

**Files:**
- Modify: `README.md`
- Create: `reproducibility/paper_v4/README.md`

**Interfaces:**
- Consumes: verified commands and real repository paths only.
- Produces: anonymous navigation, a minimal quick check, reproduction commands, extended/legacy scope, and known-discrepancy links.

- [ ] **Step 1: Rewrite README with neutral opening and real Contents**

Do not list TCN or selective metrics in Contents. Do not claim they are absent; link `implementation_scope.json` for their status. Replace the old budget quick check with:

```bash
python reproducibility/paper_v4/quick_check.py
```

- [ ] **Step 2: List only commands that pass**

Include the paper_v4 tests, unchanged historical package tests, and no moved root script path. Use `/tree/` for directory links.

- [ ] **Step 3: Add Known discrepancies and scope statements**

Point to `reproducibility/paper_v4/audits/`, preserve the CMeRT accuracy mismatch, and distinguish extended and legacy material.

- [ ] **Step 4: Add the package README and commit**

```bash
git add README.md reproducibility/paper_v4/README.md
git commit -m "docs: align repository navigation"
```

### Task 6: Validate metadata, hashes, anonymity, and push the branch

**Files:**
- Create: `reproducibility/paper_v4/SHA256SUMS.txt`
- Modify externally: GitHub repository description and topics

**Interfaces:**
- Consumes: final branch tree and GitHub repository settings.
- Produces: verified checksum manifest, neutral metadata, pushed branch URL, and final change report.

- [ ] **Step 1: Preserve and validate citation metadata**

Parse `CITATION.cff` as YAML and compare its `authors` node with `git show main:CITATION.cff`. Expected: exact structural equality and no file diff.

- [ ] **Step 2: Generate and verify SHA256SUMS**

Hash every public file below `reproducibility/paper_v4/` except `SHA256SUMS.txt`, use lowercase SHA-256 and repository-relative package paths, then independently verify every line.

- [ ] **Step 3: Run final verification**

Run:

```bash
python -m pytest -q
python reproducibility/paper_v4/quick_check.py
git diff --check main...HEAD
```

Also verify historical `paper_v2` and `paper_v3` tree hashes against `main`, scan tracked files for forbidden binary suffixes, validate every JSON/CSV, confirm Figure 3 source/copy digests, and scan new docs plus commit subjects for identifying metadata.

- [ ] **Step 4: Update neutral GitHub metadata**

Set description to `Auditable protocols, tests, and frozen summaries for causal online action detection reliability.` Set topics to `online-action-detection`, `video-understanding`, `temporal-smoothing`, `calibration`, `reproducibility`, and `reliability`.

- [ ] **Step 5: Commit checksums and push branch**

```bash
git add reproducibility/paper_v4/SHA256SUMS.txt
git commit -m "repro: seal versioned package checksums"
git push -u origin repro/paper-v4-main-evidence
```

- [ ] **Step 6: Report exact completion data**

Report:

- migrated legacy file count and manifest;
- dependency-retained file count and exact list;
- final pytest count versus baseline 148, decomposed into removed legacy tests and added paper_v4 tests;
- Figure 3 source digest, copied digest, and equality status;
- all commits, branch URL, GitHub metadata, hash verification, binary-exclusion result, and the fact that no merge or pull request was created.
