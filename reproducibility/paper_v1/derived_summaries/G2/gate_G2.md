# G2 diagnostic nonredundancy gate

- Decision: **FAIL_DOWNGRADE_PTSM**
- Frozen prediction cache: `outputs\researchwrite\stable-but-stale\analysis\G1\run_causal_gru_v1\frozen_predictions.npz`
- Videos: 211
- Transition events at each horizon/alpha: 5932
- Bootstrap: 2000 paired video-cluster resamples, seed 48624

## Primary equal-delay test

This test retains only the same ground-truth transitions for which raw and EMA 0.50 have exactly the same detected argmax delay. Any remaining paired PTSM difference cannot be encoded by a delay difference for those events.

- Eligible events: 2819 from 202 videos
- Mean paired PTSM difference: **-0.000477**
- 95% CI: **[-0.001439, 0.000512]**
- Fraction of eligible events with positive PTSM difference: 0.1391
- Primary test: **FAIL**

## Transition-type check

| Type | Events | PTSM delta | 95% CI | Delay delta | 95% CI |
|---|---:|---:|---:|---:|---:|
| background_to_action | 2921 | 0.002051 | [0.001022, 0.003129] | 0.301267 | [0.232088, 0.382380] |
| action_to_background | 2938 | 0.009256 | [0.007586, 0.010875] | 0.708645 | [0.602078, 0.809288] |
| action_to_action | 73 | 0.001933 | [-0.004574, 0.010794] | 0.986301 | [0.102032, 2.207716] |

- Passing types: background_to_action, action_to_background
- Transition-type test: **PASS**

## Horizon sensitivity

| Horizon | PTSM delta | 95% CI | Delay delta | Miss delta |
|---:|---:|---:|---:|---:|
| 8 | 0.011431 | [0.010033, 0.012754] | 0.281861 | 0.036750 |
| 16 | 0.005618 | [0.004603, 0.006612] | 0.511463 | 0.026972 |
| 32 | 0.002259 | [0.001311, 0.003162] | 0.908294 | 0.023432 |

- Horizon test: **PASS**

## ECE-bin sensitivity

| Bins | Raw ECE | EMA 0.50 ECE | Delta | 95% CI |
|---:|---:|---:|---:|---:|
| 10 | 0.118834 | 0.110589 | -0.008246 | [-0.010048, -0.007163] |
| 15 | 0.118956 | 0.110328 | -0.008628 | [-0.010033, -0.007363] |
| 20 | 0.118847 | 0.110591 | -0.008257 | [-0.010032, -0.007208] |
| 30 | 0.118956 | 0.110589 | -0.008367 | [-0.010042, -0.007247] |

- ECE-bin test: **PASS**

## Interpretation boundary

The primary exact-delay test failed. PTSM is therefore retained only as a descriptive old-versus-new probability margin and does not carry a nonredundancy or metric contribution claim. Delay and missed-transition rate carry the transition-responsiveness conclusion.
