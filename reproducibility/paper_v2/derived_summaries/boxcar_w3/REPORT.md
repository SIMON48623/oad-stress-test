# Predeclared causal boxcar W=3 replication

The only added postprocessor is the three-timestep causal boxcar average. Its mean kernel age and squared-weight sum both match the primary EMA with alpha=0.50. No other window was evaluated.

| Instance | Delta accuracy | Delta ECE | Delta TFI | Delta delay | Delta miss rate | Full inversion |
|---|---:|---:|---:|---:|---:|---|
| thumos14_causal_gru | +0.000251 | -0.004912 | -1.059983 | +0.542819 | +0.024949 | PASS |
| thumos14_linear_probe | +0.012459 | -0.043036 | -3.727074 | +0.772421 | +0.043156 | PASS |
| ek100_causal_gru | +0.001756 | -0.048185 | -1.589875 | +1.064968 | +0.061545 | PASS |

## Paired 95% cluster-bootstrap intervals

### thumos14_causal_gru

- accuracy: +0.000251 [-0.000581, +0.001119]
- global_ece: -0.004912 [-0.006071, -0.003803]
- mean_tfi: -1.059983 [-1.206691, -0.923282]
- mean_predicted_switches: -33.502370 [-39.260782, -28.189336]
- mean_transition_delay: +0.542819 [+0.481183, +0.603298]
- missed_transition_rate: +0.024949 [+0.019870, +0.030368]

### thumos14_linear_probe

- accuracy: +0.012459 [+0.010888, +0.014031]
- global_ece: -0.043036 [-0.045772, -0.040044]
- mean_tfi: -3.727074 [-4.266939, -3.237248]
- mean_predicted_switches: -107.360190 [-123.929739, -92.325000]
- mean_transition_delay: +0.772421 [+0.679824, +0.868239]
- missed_transition_rate: +0.043156 [+0.035875, +0.050448]

### ek100_causal_gru

- accuracy: +0.001756 [+0.000237, +0.003159]
- global_ece: -0.048185 [-0.051321, -0.044973]
- mean_tfi: -1.589875 [-1.751501, -1.444725]
- mean_predicted_switches: -283.548872 [-344.770489, -227.041729]
- mean_transition_delay: +1.064968 [+0.998346, +1.127133]
- missed_transition_rate: +0.061545 [+0.057343, +0.065481]

## Decision rule

A full replication requires accuracy noninferiority, significant ECE and TFI improvement, and significant worsening of both transition delay and missed-transition rate. A failed direction is reported as a boundary result and does not authorize another window.
