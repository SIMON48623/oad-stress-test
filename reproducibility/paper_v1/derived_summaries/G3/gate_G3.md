# G3 analytical counterexample gate

- Decision: **PASS**
- Exact center construction: **PASS**
- Interval-box proof: **PASS**
- ECE-bin numerical sensitivity: **PASS**
- 625-point neighborhood cross-check: **PASS**

| Bins | Raw ECE | EMA ECE | Raw minus EMA |
|---:|---:|---:|---:|
| 10 | 0.411000 | 0.165250 | 0.245750 |
| 15 | 0.411000 | 0.184250 | 0.226750 |
| 20 | 0.411000 | 0.165250 | 0.245750 |
| 30 | 0.411000 | 0.184250 | 0.226750 |

- Accuracy: raw 0.800, EMA 0.800
- Delay: raw 0, EMA 1
- Guaranteed 15-bin ECE improvement throughout the parameter box: at least 0.2236133166377496

Interpretation: the analytical gate establishes possibility and mechanism only. Empirical prevalence remains supported by G1, not by this five-frame construction.
