# Expanded postprocessor and checkpoint audit

This package extends the original audit in `reproducibility/paper_v1/` in two specific ways:

1. it evaluates a predeclared three-timestep causal boxcar average as a second postprocessor; and
2. it evaluates the primary EMA and the boxcar average on the independently released TeSTra Laplace checkpoint using the official EPIC-KITCHENS-100 validation split.

The contribution remains an evaluation result, not a new detector. The scripts compare each native probability sequence with its own causally postprocessed sequence. Absolute recognition scores are not compared across predictor families.

## What is included

- self-contained analysis scripts and protocol-logic tests;
- frozen, path-free protocol snapshots;
- small derived point estimates and paired video-cluster bootstrap intervals;
- the official TeSTra reproduction gate and its outcome;
- SHA-256 checksums for every file in this package.

## What is excluded

The package does not contain third-party source trees, checkpoints, RGB/flow features, target arrays, frozen probability caches, bootstrap draw arrays, videos, manuscripts, figures, local paths, or credentials.

## Main result and boundary

EMA with alpha 0.50 satisfies the complete ranking-inversion gate for four predictor-dataset instances: THUMOS14 causal GRU, THUMOS14 linear probe, EPIC-KITCHENS-100 causal GRU, and TeSTra on the official EPIC-KITCHENS-100 validation split.

The boxcar average reproduces the calibration/stability versus responsiveness opposition in all four instances. It also satisfies the accuracy non-inferiority gate in the first three. On TeSTra, accuracy decreases by 0.003001 with a 95% interval of [-0.004306, -0.001757], crossing the frozen non-inferiority margin of -0.002. This result is retained as a scope boundary; the window was fixed at three timesteps before inspection and no other setting was evaluated.

## Minimal verification

```bash
python -m pip install -r reproducibility/paper_v2/requirements.txt
python -m pytest -q reproducibility/paper_v2/tests
```

## Complete replay

### Boxcar audit

`postprocessors/causal_boxcar_replication.py` consumes the same frozen prediction sequences used by the original paper analyses. Run `--help` for required explicit paths. The script writes only derived summaries unless the caller chooses a different output policy.

### TeSTra checkpoint audit

1. Clone the [official TeSTra repository](https://github.com/zhaoyue-zephyrus/TeSTra) into the ignored path `reproducibility/paper_v2/vendor/TeSTra/`. The checkout is local-only and must not be committed.
2. Obtain the official Laplace checkpoint and EPIC-KITCHENS-100 RGB, TV-L1 optical-flow, and per-frame verb-target assets from the links documented by TeSTra.
3. Keep the checkpoint and data outside the tracked repository, then pass them with `--checkpoint` and `--data-root`.
4. Run `testra_ek100/run_testra_ek100.py --help` for the complete interface. The frozen protocol is loaded from `protocols/testra_ek100_protocol.json` unless `--protocol` is supplied explicitly.

The paper-facing audit is admissible only if the official 1 s mean top-5 verb-recall gate passes within 0.5 percentage point of the published 30.8% value.

## Interpretation rule

Do not tune a smoother after reading these outputs. A failed direction is a boundary result, not permission to substitute a new checkpoint, split, horizon, alpha, or window.
