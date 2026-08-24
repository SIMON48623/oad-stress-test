# OAD Stress Test

Official code and reproducibility materials for **When Stability Misleads: Transition-Conditioned Reliability in Causal Video Recognition**.

The project studies a reliability failure in online action detection: temporal smoothing can improve aggregate accuracy, calibration, and apparent stability while delaying or missing action transitions. The contribution is an evaluation protocol and diagnostic analysis rather than a new detector.

## Contents

- causal streaming evaluation code;
- prototype, linear-probe, causal GRU, and causal TCN predictors;
- transition delay, missed-transition, calibration, fragmentation, and selective-reliability metrics;
- THUMOS14 and EPIC-KITCHENS-100 data adapters;
- frozen analysis protocols, tests, checksums, and processed aggregate statistics;
- a synthetic-data workflow for installation checks.

The original THUMOS14 analysis package is in [`reproducibility/paper_v1/`](reproducibility/paper_v1/). The expanded audit with a second causal postprocessor and an official TeSTra checkpoint is in [`reproducibility/paper_v2/`](reproducibility/paper_v2/).

## Expanded evidence

The current release separates the main finding from its scope boundary.

- **EMA, alpha = 0.50:** the complete aggregate-versus-transition ranking inversion is reproduced for four predictor-dataset instances, including the independently released TeSTra Laplace checkpoint on the official EPIC-KITCHENS-100 validation split.
- **Causal boxcar, window = 3:** calibration and fragmentation improve while delay and missed-transition rate worsen in all four instances. Accuracy remains non-inferior for the two THUMOS14 predictors and the independently trained EPIC-KITCHENS-100 GRU, but not for TeSTra. The failed TeSTra accuracy gate is retained as a boundary result.
- **Official-checkpoint gate:** TeSTra's reported 1 s mean top-5 verb recall is reproduced as 30.770%, compared with the published 30.8%.

No checkpoint, third-party feature, target array, probability cache, bootstrap draw array, or manuscript is stored in this repository.

## Installation

Python 3.9 or later is required.

```bash
git clone https://github.com/SIMON48623/oad-stress-test.git
cd oad-stress-test
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

PyTorch is optional for the core evaluation package. It is required for the causal GRU and TCN predictors and for model-based manuscript replays.

## Quick check

The following commands use generated synthetic data and do not require a research dataset.

```bash
python scripts/prepare_dummy_data.py --config configs/dummy.yaml
python scripts/run_baseline.py --config configs/dummy.yaml --policy uniform --budgets 0.10 0.25 0.50 1.00 --overwrite
python scripts/run_baseline.py --config configs/dummy.yaml --policy random --budgets 0.10 0.25 0.50 1.00 --overwrite
python scripts/evaluate.py --results-dir results/dummy --policies uniform random --budgets 0.10 0.25 0.50 1.00 --clean-summary
python -m pytest -q
```

## Research datasets

THUMOS14, EPIC-KITCHENS-100, pretrained features, and model artifacts are not redistributed. Obtain them from their original providers and follow [`DATASET.md`](DATASET.md) for the expected local layout.

## Reproducing the paper analyses

The frozen packages contain the original G1-G4 analyses and the expanded postprocessor/checkpoint audit.

```bash
python -m pip install -r reproducibility/paper_v1/requirements.txt
python -m pytest -q reproducibility/paper_v1/tests
python -m pytest -q reproducibility/paper_v2/tests
```

A complete model replay additionally requires authorized feature files and, where specified by a protocol, the corresponding checkpoint or frozen prediction sequence. See the README in the relevant reproducibility package.

## Repository layout

```text
configs/              experiment configurations
reproducibility/      frozen paper analysis package
scripts/              preparation, training, evaluation, and plotting entry points
src/oad_stress_test/  installable Python package
tests/                unit and pipeline tests
tools/                transition and reliability analysis utilities
```

## Scope

1. Every online decision is causal: no future feature, label, or prediction is available at time step `t`.
2. The reported resource constraint is a feature-level observation or inference budget, not a hardware-deployment claim.
3. Wait and abstain policies and lightweight predictors are diagnostic baselines, not state-of-the-art methods.
4. Aggregate statistics in this repository are released for audit; third-party source data remain governed by their original terms.

## Citation

Use the metadata in [`CITATION.cff`](CITATION.cff). The article DOI will be added after publication.

## License

Original source code is released under the [MIT License](LICENSE). Third-party datasets, annotations, features, and model artifacts are not covered by this licence.
