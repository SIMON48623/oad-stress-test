# G4 second-predictor replication gate

- Decision: **PASS_PRIMARY**
- Gate-carrying alpha: **0.5**
- Predictor: existing THUMOS14 multinomial linear probe
- Videos / frames / transitions: 211 / 334123 / 5932
- Selected original-interface equivalence max absolute difference: 0.000e+00 (**PASS**)
- Rejected batch candidate max absolute difference: 2.980e-07
- Bootstrap: 2000 paired video-cluster resamples, seed 48625

## Point estimates

| Alpha | Accuracy | Global ECE | TFI | Delay | Miss rate | PTSM (descriptive) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.656686 | 0.189560 | 7.666833 | 4.772758 | 0.187626 | 0.210089 |
| 0.25 | 0.662023 | 0.167579 | 6.095608 | 5.005394 | 0.201113 | 0.204649 |
| 0.50 | 0.671373 | 0.138025 | 4.399182 | 5.421443 | 0.226062 | 0.200769 |

## Paired differences (smoothed minus raw)

| Alpha | Metric | Delta | 95% CI |
|---:|---|---:|---:|
| 0.50 | accuracy | 0.014686 | [0.013002, 0.016359] |
| 0.50 | global_ece | -0.051534 | [-0.054652, -0.048055] |
| 0.50 | mean_tfi | -3.267651 | [-3.748054, -2.839725] |
| 0.50 | mean_transition_delay | 0.648685 | [0.555115, 0.745142] |
| 0.50 | missed_transition_rate | 0.038436 | [0.031145, 0.045640] |
| 0.50 | ptsm | -0.009320 | [-0.011599, -0.007190] |
| 0.25 | accuracy | 0.005336 | [0.004648, 0.006051] |
| 0.25 | global_ece | -0.021980 | [-0.023344, -0.020567] |
| 0.25 | mean_tfi | -1.571225 | [-1.823897, -1.345056] |
| 0.25 | mean_transition_delay | 0.232637 | [0.184311, 0.283782] |
| 0.25 | missed_transition_rate | 0.013486 | [0.010028, 0.017002] |
| 0.25 | ptsm | -0.005440 | [-0.006474, -0.004483] |

## Gate checks

### Alpha 0.50

- global_ece_improves: **PASS**
- tfi_improves: **PASS**
- accuracy_noninferior: **PASS**
- delay_worsens: **PASS**
- miss_rate_worsens: **PASS**
- responsiveness_worsens: **PASS**
- overall: **PASS**

### Alpha 0.25

- global_ece_improves: **PASS**
- tfi_improves: **PASS**
- accuracy_noninferior: **PASS**
- delay_worsens: **PASS**
- miss_rate_worsens: **PASS**
- responsiveness_worsens: **PASS**
- overall: **PASS**

## Leave-one-video-out direction ranges

- alpha 0.50: ECE delta [-0.051942, -0.050967]; delay delta [0.617604, 0.665386].
- alpha 0.25: ECE delta [-0.022176, -0.021779]; delay delta [0.212886, 0.238736].

PTSM is reported descriptively and does not carry this gate. Passing supports replication across two fitted predictors on THUMOS14, not cross-dataset generality.
