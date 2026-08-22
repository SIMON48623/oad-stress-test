# Minimal analytical ranking-inversion counterexample

## Construction

Let the binary ground truth be `y=(0,0,1,1,1)` and let `p_t` denote the raw probability of class 1. Choose

`p=(0.04, 0.71, 0.565, 0.565, 0.565)` and causal EMA `s_t = 0.5 s_{t-1} + 0.5 p_t`, initialized with `s_0=p_0`.

The exact smoothed sequence is

`s=(1/25, 3/8, 47/100, 207/400, 433/800) = (0.040000, 0.375000, 0.470000, 0.517500, 0.541250)`.

The raw predictor makes one isolated pre-transition error and then reacts at the transition immediately. EMA removes that isolated error but transfers the error to the first post-transition frame.

## Exact outcome

- Raw accuracy: 4/5 = 0.800000
- EMA accuracy: 4/5 = 0.800000
- Raw transition delay: 0
- EMA transition delay: 1
- Raw 15-bin ECE: 411/1000 = 0.411000
- EMA 15-bin ECE: 737/4000 = 0.184250

Thus accuracy is unchanged, global ECE improves, and transition delay worsens.

## Non-singleton parameter region

Let `l∈[0.039,0.041]`, `g∈[0.709,0.711]`, `h∈[0.564,0.566]`, and `a∈[0.499,0.501]`, with raw sequence `(l,g,h,h,h)` and EMA coefficient `a`. Interval propagation gives:

- s1: [0.373252, 0.376752]
- s2: [0.467688748, 0.472318752]
- s3: [0.514812685252, 0.520197694752]
- s4: [0.538327529940748, 0.544185045070752]

Throughout the box, `s1<0.5`, `s2<0.5`, and `s3>0.5`. Therefore EMA always corrects the isolated old-segment spike and always responds one frame later than the raw predictor.

Under the fixed 15-bin assignments,

`ECE_raw = [l + g + 3(1-h)]/5`,

`ECE_EMA = [l + s1 + (s3-s2) + (1-s4)]/5`.

Interval bounds give `ECE_raw ∈ [0.410, 0.412]` and `ECE_EMA ∈ [0.1821121776362496, 0.1863866833622504]`. The guaranteed gap is at least 0.2236133166377496.

## Scope boundary

This construction proves existence, not inevitability. It formalizes how aggregate calibration can reward suppression of an isolated fluctuation while ignoring that the same causal memory delays a true change.
