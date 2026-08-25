# Dataset preparation

This repository does not redistribute videos, annotations, pretrained features, checkpoints, or prediction arrays from third parties. Download each resource from its original provider and keep it outside version control.

## THUMOS14

Obtain the dataset and temporal annotations from the [official THUMOS14 download page](https://www.crcv.ucf.edu/THUMOS14/download.html). Obtain the I3D representation used by a replay from its original provider. The default configuration expects:

```text
data/thumos14/
  annotations/
    thumos14.csv
  features/
    <video_id>.npy
  splits/
    train.txt
    test.txt
```

`thumos14.csv` must contain the columns `video_id`, `start_idx`, `end_idx`, and `label`. Indices refer to the temporal steps of the local feature representation, not raw video frame numbers. Each split file contains one video identifier per line.

After preparing the files, run:

```bash
python scripts/check_thumos_ready.py --config configs/thumos14.yaml
python scripts/prepare_data.py --config configs/thumos14.yaml
```

The frozen paper protocols record the expected conventions and hashes for the experiment inputs. Do not substitute a feature stride or annotation-to-feature conversion without documenting the change.

### CMeRT checkpoint replay

The CMeRT audit in `reproducibility/paper_v3/cmert_thumos/` uses the authors' THUMOS14 architecture and configuration, a hash-pinned epoch-9 checkpoint, and the TeSTra-distributed RGB, TV-L1 flow, and per-frame target arrays expected by CMeRT. The published action-detection and mean-anticipation per-frame mAP values are reproduced as 0.73221 and 0.59442.

Keep the third-party assets outside this repository. Their archive/checkpoint hashes and the 213-session shape contract are recorded in `reproducibility/paper_v3/protocols/cmert_thumos_protocol.json`. Export only the current-time probability vector and matching target vector for each official test video, then run the output-level analysis described in `reproducibility/paper_v3/README.md`.

## EPIC-KITCHENS-100

Access EPIC-KITCHENS-100 through the [official dataset portal](https://epic-kitchens.github.io/2021). The repository accepts locally prepared target arrays and RGB/flow feature files; use `scripts/inspect_ek100_replication_inputs.py` to verify their shapes and alignment before a run.

The TeSTra audit in `reproducibility/paper_v2/testra_ek100/` uses the official TeSTra repository, its released Laplace checkpoint, the official RGB and TV-L1 optical-flow features, and official per-frame verb targets. These assets remain under their original providers' terms and must be stored outside this repository. The frozen protocol records expected file hashes, dimensions, split counts, and the exact current-output definition.

## TVSeries

TVSeries is not redistributed. The public code includes schema and transition-audit utilities, but no dataset files or claim of a completed TVSeries replication.

## Local-data rule

Keep all downloaded or generated research data under ignored local directories. Before publishing a fork, inspect the tracked file list and Git history rather than relying only on `.gitignore`.
