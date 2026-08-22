# Supplementary reproducibility package

This package contains the manuscript-specific analysis scripts, frozen protocols, tests, and derived summary statistics for `When Stability Misleads: Transition-Conditioned Reliability in Causal Video Recognition`. It deliberately excludes the manuscript, figures, bibliography, copyrighted videos, third-party I3D feature files, trained checkpoints, frozen prediction arrays, and bootstrap draw arrays.

## Public framework dependency

- Repository: https://github.com/SIMON48623/oad-stress-test
- Archival release: `v1.0-transition-reliability`

Clone the archival release, install it in an isolated Python environment, and install this package's `requirements.txt`. The G1 and G4 scripts accept explicit project-source, checkpoint, feature, split, annotation, protocol, and output paths. G2 consumes the frozen G1 prediction cache. G3 is data-independent and verifies the analytical construction.

The THUMOS14 videos and ActionFormer-distributed I3D features must be obtained from their original sources. Checkpoint hashes and frozen design choices are retained in the protocol files; local machine paths have been replaced by `${PROJECT_ROOT}` placeholders.

## Minimal verification

Run `pytest -q` from this directory. To run a complete scientific replay, follow the command-line help of each script and supply the licensed external data and local checkpoint files.

## Scope

The package reproduces the calculations reported in the manuscript from the required model probability sequences. It does not claim to redistribute third-party data or to recreate the original pretrained feature extractor.
