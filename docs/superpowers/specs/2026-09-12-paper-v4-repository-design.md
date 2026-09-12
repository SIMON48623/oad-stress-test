# Paper v4 Repository Design

## Purpose

Reorganize the repository so that one anonymous manuscript version has a single, auditable evidence package while all historical, extended, discrepant, and legacy materials remain preserved. The work adds no experimental results and does not alter any frozen numeric value.

## Non-negotiable constraints

- Do not delete frozen experiment data, bootstrap outputs, provenance records, or audit files.
- Preserve numeric discrepancies as evidence; never edit a value to force agreement.
- Do not create new experimental result files.
- Keep `reproducibility/paper_v2/` and `reproducibility/paper_v3/` byte-for-byte unchanged.
- Keep the author records in `CITATION.cff` unchanged.
- Do not place manuscript titles, venue names, author names, or affiliations in the root README, newly added documentation, or commit messages.
- Do not publish checkpoints, prediction caches, or bootstrap draw arrays. Record their external source paths, sizes, and SHA-256 digests in provenance only.
- Keep `src/oad_stress_test/models/causal_tcn.py` and `src/oad_stress_test/metrics/selective.py` in the main source tree. They have implementations and tests, but the main manuscript does not depend on them.
- Complete the work on `repro/paper-v4-main-evidence`; push that branch only. Do not merge it into `main` and do not create a pull request.

## Evidence architecture

Create `reproducibility/paper_v4/` as an additive evidence layer. Existing versioned packages remain historical sources and are referenced by repository-relative paths and SHA-256 digests.

The package contains:

```text
reproducibility/paper_v4/
  README.md
  SHA256SUMS.txt
  quick_check.py
  derived_summaries/
    ek100_causal_gru_ema/
      point_estimates.csv
      bootstrap_deltas.csv
      provenance.json
  reproduction_logs/
    testra_ek100.json
    cmert_thumos14.json
  protocols/
    instance_splits_and_counts.json
    training_config.json
    bootstrap_protocol.json
    metric_protocol.json
  figure3_source/
    figure3_transition_recovery.csv
    figure3_audit.json
    provenance.json
  audits/
    five_instance_boxcar_ci.csv
    cmert_boxcar_accuracy_mismatch.json
    displayed_value_trace.json
    implementation_scope.json
  tests/
    test_quick_check.py
    test_metric_protocol.py
    test_boxcar_ci.py
    test_package_integrity.py
```

Only existing frozen rows are copied into the derived summaries and audit tables. Copying a frozen result into the versioned package is a provenance-preserving relocation, not a recomputation. Every copied file records its source path and source digest.

## Recovered EK100 EMA package

The recovered EK100 causal-GRU EMA package is sourced from the separately retained recovery snapshot. The public package includes only `point_estimates.csv` and `bootstrap_deltas.csv`; it excludes `trained_model.pt`, `frozen_predictions.npz`, and `bootstrap_samples.npz`.

`provenance.json` records:

- the original server-side logical path and the dated local recovery snapshot identifier;
- recovery date `2026-09-12`;
- byte size and SHA-256 for every public derived file and every excluded large artifact that is present in the recovery snapshot;
- the declared and observed `formal_run.log` checksums;
- the located checksum-mismatch cause: the checksum list was created while the log was still being appended through `tee`, and the completion marker was written after the digest was taken;
- a status that preserves the mismatch instead of normalizing it.

## Reproduction logs and protocols

The TeSTra record preserves the published-metric reproduction of `30.76995422935509%` and records its existing source path, digest, sequence count, timestep count, transition count, and validation-segment count.

The CMeRT record preserves detection mAP `73.221` and anticipation mAP `59.442`, with its existing source path, digest, video count, valid-output count, and transition count.

The protocol inventory records all five manuscript instances. It includes split membership or a split-manifest reference, the two excluded THUMOS14 video identifiers, the EK100 `500/133` train/evaluation split, per-instance timesteps, primary transitions, clean transitions, and any source-declared sample count. Missing fields are not guessed: the record uses an explicit `not_reported_in_frozen_source` status and names the searched source files.

Training and inference settings include training seed `13`, EK100 EMA bootstrap seed `48627`, CMeRT boxcar bootstrap seed `48629`, EMA alpha `0.50`, horizon `H=16`, and all source-declared optimizer and architecture settings. The bootstrap protocol fixes 2,000 paired resamples clustered by video or session, as appropriate for each instance.

## Metric-contract tests

Tests exercise observable protocol behavior using small hand-checked fixtures. They cover:

- active-set top-1 selection;
- clean-transition filtering;
- horizon `H=16`;
- truncation at the next true transition;
- assigning `H` to a miss;
- the expected sign of the boxcar confidence interval for ECE, delay, and missed-transition rate in all five instances;
- integrity and provenance references for the versioned package.

The tests do not rerun training, bootstrap sampling, or manuscript experiments.

## CMeRT mismatch audit

`cmert_boxcar_accuracy_mismatch.json` contains at least these fields:

- `source_exact`: `-0.007333105779500304`;
- `correct_4dp`: `-0.0073`;
- `manuscript_display`: `-0.0077`;
- `audit_status`: a value that explicitly preserves an unresolved mismatch;
- `suspected_origin`: `-0.007723984541020301` from the separate active-set audit;
- `suspected_origin_status`: `unverified_hypothesis_not_an_explanation`;
- source paths and SHA-256 digests for both records.

The active-set value is recorded only as a hypothesis requiring verification. It is not presented as the source of the manuscript value and is not used to reconcile the mismatch.

`displayed_value_trace.json` records that `-0.0018`, `+0.9838`, and `+0.0331` have matching frozen boxcar sources and are ordinary four-decimal displays.

## Figure 3 source

Copy the existing 64-row Figure 3 transition-recovery CSV and its audit JSON into version control without numeric edits. Record row count, instance names, offsets, transition counts, source paths, and SHA-256 in a companion provenance file. The audit requires the copied file digests to match the previously identified source digests.

## Extended analyses

Create `reproducibility/extended/README.md` as an index rather than moving files out of historical packages. It identifies TFI, predicted-switch counts, alpha `0.25`, ECE bin sweeps, `H=8/32`, leave-one-out validation, and all-transition analyses as outside the main manuscript evidence chain. Every item links to a real retained path. Historical files remain unchanged.

## Legacy baseline

Move files whose primary purpose is the older budget, sampling-policy, wait, or abstain baseline into `legacy/budget_policy_wait_abstain/`. Preserve file contents and original relative paths under that directory whenever possible. Move dedicated tests and documentation with the corresponding code, and keep a manifest mapping every original path to its destination.

Classify files by primary interface, not by keyword alone:

- move policy implementations, budget-specific metrics, baseline entry points, budget/policy plots, wait/abstain analysis tools, dedicated configurations, and tests whose behavior depends on those interfaces;
- retain shared dataset, model, transition, ECE, and provenance utilities used by the manuscript evidence packages;
- retain `causal_tcn.py` and `metrics/selective.py` in the main source tree;
- record `implementation_present_main_manuscript_not_dependent` for those two retained modules in `audits/implementation_scope.json`;
- if moving one file would make a retained module or retained test unloadable, retain the shared file and document the dependency decision instead of creating a compatibility shim.

Legacy tests are preserved as historical baseline tests. The root CI continues to run the supported main-tree and reproducibility tests; the legacy README states the original commit needed to replay the historical baseline if its old import layout is required.

## Root README and quick check

Rewrite the root README with neutral wording and no identifying manuscript metadata. Its Contents section lists only real paths and distinguishes the main evidence package, historical packages, extended analyses, current implementation, and legacy baseline.

The quick check is the only new executable production code in this change. It uses deterministic synthetic probabilities and labels, performs native and causal EMA evaluation, finishes in seconds, writes no result files, and prints accuracy, ECE, mean transition delay, and missed-transition rate for both variants. Development follows test-first red/green/refactor cycles.

The README includes only commands verified in the final worktree. Directory links use repository-relative links or GitHub `/tree/` links, never `/blob/` links for directories. A `Known discrepancies` section points to `reproducibility/paper_v4/audits/` and summarizes status without changing any value.

## Metadata and release checks

- Validate `CITATION.cff` as YAML/CFF and prove its `authors` value is unchanged from `main`.
- Verify GitHub Actions runs pytest and that the final local command matches the workflow's behavior.
- Audit tracked files against `.gitignore` and the excluded-artifact policy.
- Set a neutral repository description and relevant topics without manuscript, venue, author, or affiliation identifiers.
- Scan the root README, all newly added documentation, and all new commit messages for identifying metadata.
- Generate `reproducibility/paper_v4/SHA256SUMS.txt` from the final public package, excluding the checksum file itself, and verify every entry.

## Completion and reporting

Each implementation checkpoint reports what changed, why it changed, and the exact source files supporting the change. Final verification includes the complete pytest suite, the standalone quick check, checksum verification, frozen-source comparison, author-field comparison, ignored-artifact audit, anonymity scan, and `git diff --check`.

After verification, commit and push `repro/paper-v4-main-evidence`. Report the branch URL and change summary. Do not merge into `main` and do not create a pull request.
