# Paper v3: CMeRT checkpoint validation

This frozen package adds a CMeRT–THUMOS14 checkpoint audit to the four predictor–dataset instances in `paper_v2`. It contains the protocol, executable output-level audit, tests, and lightweight derived summaries used by the revised manuscripts.

## Package contents

- exact reproduction of 0.73221 action-detection per-frame mAP and 0.59442 mean anticipation mAP;
- 213 official THUMOS14 test videos, 181,642 valid feature timesteps, and 5,253 clean transitions;
- EMA (`alpha = 0.50`) and an additional causal boxcar (`W = 3`) comparison;
- 2,000 paired video-level bootstrap resamples with seed 48629.

The checkpoint, features, targets, emitted probability arrays, per-video statistics, and bootstrap draw arrays are deliberately excluded. Their hashes or aggregate summaries are recorded where redistribution permits.

## Re-run from exported probabilities

First run the CMeRT repository using its THUMOS14 configuration and checkpoint. At the end of its test loop, export each session's `pred_scores[session]` and `gt_targets[session]` as:

```python
np.savez_compressed(
    output_dir / f"{session}.npz",
    probabilities=np.asarray(pred_scores[session], dtype=np.float32),
    targets=np.asarray(gt_targets[session], dtype=np.float32),
)
```

The export directory must contain exactly 213 files named `video_test_*.npz`. Then run:

```bash
python reproducibility/paper_v3/cmert_thumos/run_cmert_thumos.py \
  --input-dir /path/to/cmert_probability_exports \
  --output-dir /path/to/audit_output
python -m pytest -q reproducibility/paper_v3/tests/test_cmert_thumos.py
```

The runner validates shapes and normalized probabilities before computing results. See `protocols/cmert_thumos_protocol.json` for the frozen parameters and `derived_summaries/cmert_thumos/` for the expected aggregate outputs.

## Interpretation

Under EMA, CMeRT substantially reduces ECE, TFI, and predicted switches, while delay and missed-transition rate increase. Accuracy decreases by 0.00352. This extends the central stability-responsiveness conflict to a strong 2025 checkpoint. The supported explanation is output-level temporal inertia across a true action boundary.
