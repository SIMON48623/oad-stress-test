# CMeRT–THUMOS14 validation report

The published CMeRT action-detection and mean-anticipation mAP values were reproduced as 0.73221 and 0.59442 before the output audit.

With EMA (`alpha = 0.50`), global ECE fell by 0.0098 and mean TFI by 0.3692. Mean predicted switches fell from 50.0 to 34.1 per video. At the same time, transition delay increased by 0.8300 feature timesteps (0.208 s) and the missed-transition rate rose by 0.0335. Accuracy changed by -0.00352.

These results extend the calibration/fragmentation-versus-transition conflict to a strong CVPR 2025 predictor. The supported mechanism is output-level temporal inertia, in which causal smoothing carries probability mass from the preceding class across an action boundary.
