# TeSTra-EK100 frozen checkpoint audit

## Official checkpoint integrity gate

- Published 1.0 s overall mean top-5 verb recall: 30.8%
- Reproduced horizons: [2.0, 1.0]
- Reproduced values: [26.234, 30.77]%
- Reproduction gate: PASS

The paper-facing OAD audit is admissible only when the official checkpoint gate passes.

## Current-timestep causal postprocessing audit

| Condition | Delta accuracy | Delta ECE | Delta TFI | Delta delay | Delta miss rate | Full inversion |
|---|---:|---:|---:|---:|---:|---|
| ema_alpha_0.50 | +0.001153 | -0.050358 | -1.296156 | +0.382778 | +0.022328 | PASS |
| boxcar_w3 | -0.003001 | -0.037352 | -1.449391 | +0.490729 | +0.024949 | FAIL |

A failed direction is a boundary result. It does not authorize changing the checkpoint, split, EMA alpha, boxcar window, horizon, or bootstrap seed.
