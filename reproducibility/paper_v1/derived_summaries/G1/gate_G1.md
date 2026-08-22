# G1 confirmatory ranking-inversion gate

- Decision: **PASS_PRIMARY**
- Gate-carrying alpha: **0.5**
- Videos: 211
- Frames: 334123
- Ground-truth transitions: 5932
- Bootstrap: 2000 paired video-cluster resamples, seed 48623

## Frozen point estimates

| EMA alpha | Accuracy | Global ECE | TFI | Switches/video | Delay | PTSM | Miss rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.718346 | 0.118956 | 2.585356 | 90.076 | 6.120533 | 0.235968 | 0.278995 |
| 0.25 | 0.719499 | 0.114664 | 2.046632 | 73.336 | 6.323331 | 0.237141 | 0.289110 |
| 0.50 | 0.720109 | 0.110328 | 1.520559 | 56.550 | 6.631996 | 0.241447 | 0.305968 |

## Paired differences (smoothed minus raw)

| Alpha | Metric | Delta | 95% CI | Expected direction probability |
| ---: | --- | ---: | ---: | ---: |
| 0.50 | accuracy | 0.001763 | [0.000804, 0.002748] | 1.0000 |
| 0.50 | global_ece | -0.008628 | [-0.010033, -0.007302] | 1.0000 |
| 0.50 | ptsm | 0.005479 | [0.004488, 0.006536] | 1.0000 |
| 0.50 | mean_transition_delay | 0.511463 | [0.454361, 0.567121] | 1.0000 |
| 0.50 | missed_transition_rate | 0.026972 | [0.021658, 0.032331] | 1.0000 |
| 0.50 | mean_tfi | -1.064797 | [-1.203663, -0.930538] | 1.0000 |
| 0.25 | accuracy | 0.001152 | [0.000696, 0.001621] | 1.0000 |
| 0.25 | global_ece | -0.004291 | [-0.004956, -0.003134] | 1.0000 |
| 0.25 | ptsm | 0.001173 | [0.000794, 0.001583] | 1.0000 |
| 0.25 | mean_transition_delay | 0.202798 | [0.165416, 0.238918] | 1.0000 |
| 0.25 | missed_transition_rate | 0.010115 | [0.007056, 0.013273] | 1.0000 |
| 0.25 | mean_tfi | -0.538724 | [-0.623906, -0.465493] | 1.0000 |

## Gate checks

### Alpha 0.50

- global_ece_improves: **PASS**
- ptsm_worsens: **PASS**
- accuracy_noninferior: **PASS**
- delay_worsens: **PASS**
- miss_rate_worsens: **PASS**
- responsiveness_worsens: **PASS**
- overall: **PASS**

### Alpha 0.25

- global_ece_improves: **PASS**
- ptsm_worsens: **PASS**
- accuracy_noninferior: **PASS**
- delay_worsens: **PASS**
- miss_rate_worsens: **PASS**
- responsiveness_worsens: **PASS**
- overall: **PASS**

## Leave-one-video-out influence

- alpha 0.50: ECE delta range [-0.008764, -0.008355]; PTSM delta range [0.005345, 0.005702].
- alpha 0.25: ECE delta range [-0.004363, -0.004164]; PTSM delta range [0.001123, 0.001267].

## Interpretation boundary

This gate tests one frozen predictor and one dataset. Passing confirms the paired empirical ranking inversion under the registered protocol; it does not establish universal generality or metric superiority.
