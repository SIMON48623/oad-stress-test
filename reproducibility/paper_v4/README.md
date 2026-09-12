# Primary reproducibility package

This directory contains the frozen, single-version evidence chain for the current manuscript. It does not rewrite or replace the historical `paper_v2` and `paper_v3` packages.

## Package contents

- `derived_summaries/ek100_causal_gru_ema/`: exact filtered point estimates and clustered-bootstrap deltas for the native and EMA 0.50 conditions, plus recovery provenance.
- `reproduction_logs/`: frozen published-metric reproductions with source paths, SHA-256 digests, and sample counts.
- `protocols/`: complete instance membership, timestep and transition counts, training configuration, metric rules, and clustered-bootstrap protocol.
- `figure3_source/`: byte-preserved CSV source data, the associated audit JSON, and copy provenance.
- `audits/`: the five-instance boxcar confidence-interval summary and the unresolved displayed-value mismatch record.
- `tests/`: executable checks for metric semantics, confidence-interval coverage and direction, provenance integrity, and excluded binary artifacts.
- `quick_check.py`: deterministic synthetic native-versus-EMA comparison that writes no files.
- `SHA256SUMS.txt`: SHA-256 manifest for every package file other than the manifest itself.

## Minimal verification

From the repository root:

```bash
python reproducibility/paper_v4/quick_check.py
python -m pytest -q reproducibility/paper_v4/tests
```

The quick check reports accuracy, expected calibration error, mean transition delay, and missed-transition rate for native and EMA outputs. It uses only in-memory synthetic arrays with 20 transitions; the EMA changes are approximately `+0.0050` accuracy, `-0.0705` ECE, `+0.8` timestep delay, and `+0.05` missed-transition rate.

## Evidence rules

The transition protocol uses active-set top-1 correctness, clean transitions with one active label on both sides, a horizon of 16 feature timesteps, truncation at the next ground-truth transition, and delay 16 for a miss. Bootstrap intervals use 2,000 video- or session-clustered resamples. Instance-specific seeds and resampling units are frozen in `protocols/bootstrap_protocol.json`.

The five-instance boxcar audit preserves the exact source intervals. Across all five instances, the calibration interval is strictly below zero while the transition-delay and missed-transition intervals are strictly above zero. Accuracy direction is reported but is not constrained to a common sign.

## Provenance and excluded artifacts

The package follows the summary-only publication policy. It does not contain `trained_model.pt`, `frozen_predictions.npz`, or `bootstrap_samples.npz`. Their recovered source paths, byte sizes, and SHA-256 digests are retained in the derived-summary provenance file.

The recovered formal-run log contains an internal checksum that differs from the recovered file digest. The provenance record preserves both values and localizes the cause: the script hashed the log while the logging pipeline was still appending, and a completion marker was written after the digest line. This is a checksum-timing issue, not a replacement of the recovered results.

## Known discrepancies

The boxcar accuracy audit preserves four separate fields: `source_exact`, `correct_4dp`, `manuscript_display`, and `audit_status`. The additional `suspected_origin` value is marked `unverified_hypothesis_not_an_explanation`; it is a lead for future checking and does not resolve the discrepancy.

Three other displayed literals have matching frozen source values and are indexed in the repository-level source trace. No numerical value is rewritten to force agreement.

## Verify checksums

On a shell with `sha256sum`:

```bash
cd reproducibility/paper_v4
sha256sum --check SHA256SUMS.txt
```

On PowerShell, compare each listed digest with `Get-FileHash -Algorithm SHA256`.
