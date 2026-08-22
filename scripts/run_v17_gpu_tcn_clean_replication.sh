#!/usr/bin/env bash
set -euo pipefail

# v1.7 formal run script.
# Run manually on a CUDA machine:
#   export OUTPUT_ROOT=/path/to/result/storage
#   chmod +x scripts/run_v17_gpu_tcn_clean_replication.sh
#   bash scripts/run_v17_gpu_tcn_clean_replication.sh
#
# Summary-only package command after completion:
#   tar -czf v17_gpu_tcn_clean_replication_summary_only.tar.gz \
#     "${ROOT}/environment.md" \
#     "${ROOT}/nvidia_smi.txt" \
#     "${ROOT}/run_config.md" \
#     "${ROOT}/atm_summary.csv" \
#     "${ROOT}/bootstrap_summary.csv" \
#     "${ROOT}"/analysis_*/atm_summary.csv \
#     "${ROOT}"/analysis_*/summary_readable.md \
#     "${ROOT}"/bootstrap_*/bootstrap_summary.csv \
#     "${ROOT}"/bootstrap_*/summary_readable.md

# Set OUTPUT_ROOT to a location with enough space for intermediate arrays.
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULTS_ROOT:-results}}"
ROOT="${ROOT:-${OUTPUT_ROOT%/}/v1.7_gpu_tcn_clean_replication}"
CONFIG="${CONFIG:-configs/thumos14.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BOOTSTRAP="${BOOTSTRAP:-1000}"
BOOTSTRAP_BACKEND="${BOOTSTRAP_BACKEND:-numpy}"
SEED="${SEED:-42}"
TCN_MAX_EPOCHS="${TCN_MAX_EPOCHS:-20}"
TCN_HIDDEN_DIM="${TCN_HIDDEN_DIM:-128}"
TCN_NUM_LAYERS="${TCN_NUM_LAYERS:-4}"
TCN_KERNEL_SIZE="${TCN_KERNEL_SIZE:-3}"
TCN_DILATIONS="${TCN_DILATIONS:-1 2 4 8}"
TCN_DROPOUT="${TCN_DROPOUT:-0.0}"
TCN_LR="${TCN_LR:-0.001}"
TCN_CHUNK_LENGTH="${TCN_CHUNK_LENGTH:-512}"
KEEP_INTERMEDIATES="${KEEP_INTERMEDIATES:-0}"

# Required by torch deterministic CUDA algorithms in some environments.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mkdir -p "${ROOT}"

echo "=== v1.7 CUDA environment check ==="
"${PYTHON_BIN}" - <<'PY' | tee "${ROOT}/environment.md"
from __future__ import annotations

import platform
import sys

try:
    import torch
except Exception as exc:  # noqa: BLE001 - clear server-side setup failure.
    raise SystemExit(f"ERROR: PyTorch import failed: {exc}") from exc

print("# v1.7 GPU Causal TCN Environment")
print()
print(f"- python: {sys.version.split()[0]}")
print(f"- platform: {platform.platform()}")
print(f"- torch.__version__: {torch.__version__}")
print(f"- torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"- torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"- torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("ERROR: v1.7 formal run requires CUDA; refusing to fall back to CPU.")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== nvidia-smi ==="
  nvidia-smi | tee "${ROOT}/nvidia_smi.txt"
else
  echo "nvidia-smi not found on PATH" | tee "${ROOT}/nvidia_smi.txt"
fi

cat > "${ROOT}/run_config.md" <<EOF
# v1.7 Causal TCN Clean Replication Run Config

- dataset: THUMOS14 clean
- model: causal_tcn
- device: cuda
- train split: clean THUMOS14 train split from configs/thumos14.yaml
- eval split: full 211-video THUMOS14 test split
- hidden_dim: ${TCN_HIDDEN_DIM}
- num_layers: ${TCN_NUM_LAYERS}
- kernel_size: ${TCN_KERNEL_SIZE}
- dilations: ${TCN_DILATIONS}
- dropout: ${TCN_DROPOUT}
- max_epochs: ${TCN_MAX_EPOCHS}
- lr: ${TCN_LR}
- chunk_length: ${TCN_CHUNK_LENGTH}
- bootstrap: ${BOOTSTRAP}
- bootstrap_backend: ${BOOTSTRAP_BACKEND}
- bootstrap_unit: video
- near_window: 8
- far_window: 32
- policy: confidence_threshold
- budgets: 0.25, 1.00
- modes:
  - max_prob thresholds: 0.50, 0.70, 0.90
  - entropy thresholds: 1.0, 2.0
  - top2_margin thresholds: 0.30, 0.50, 0.78
- per_frame_logging: enabled for analyzer input
- keep_intermediates: ${KEEP_INTERMEDIATES}
- cublas_workspace_config: ${CUBLAS_WORKSPACE_CONFIG}

The first mode should fit the causal TCN. Later modes should reuse the same
classifier cache key. If keep_intermediates is 0, per-frame JSONL and
frame-level analyzer CSV files are deleted after successful bootstrap.
EOF

run_mode() {
  local mode="$1"
  local thresholds="$2"
  local jsonl="${ROOT}/per_frame_causal_tcn_thumos14_clean_${mode}.jsonl"
  local analysis_dir="${ROOT}/analysis_${mode}"
  local bootstrap_dir="${ROOT}/bootstrap_${mode}"
  local run_log="${ROOT}/run_${mode}.log"

  echo "=== v1.7 full CUDA eval: mode=${mode} thresholds=${thresholds} ==="
  # shellcheck disable=SC2086
  "${PYTHON_BIN}" scripts/run_baseline.py \
    --config "${CONFIG}" \
    --classifier causal_tcn \
    --policy confidence_threshold \
    --budgets 0.25 1.00 \
    --thresholds ${thresholds} \
    --uncertainty-mode "${mode}" \
    --summary-only \
    --dump-per-frame \
    --per-frame-output "${jsonl}" \
    --results-dir "${ROOT}" \
    --tcn-device cuda \
    --tcn-hidden-dim "${TCN_HIDDEN_DIM}" \
    --tcn-num-layers "${TCN_NUM_LAYERS}" \
    --tcn-kernel-size "${TCN_KERNEL_SIZE}" \
    --tcn-dilations ${TCN_DILATIONS} \
    --tcn-dropout "${TCN_DROPOUT}" \
    --tcn-max-epochs "${TCN_MAX_EPOCHS}" \
    --tcn-lr "${TCN_LR}" \
    --tcn-chunk-length "${TCN_CHUNK_LENGTH}" \
    --shift-name clean \
    --overwrite \
    --progress-every 20 | tee "${run_log}"

  echo "=== validate per-frame JSONL: ${mode} ==="
  "${PYTHON_BIN}" tools/validate_per_frame_log.py --input "${jsonl}" | tee "${ROOT}/validate_${mode}.log"

  echo "=== transition misalignment analysis: ${mode} ==="
  "${PYTHON_BIN}" tools/analyze_transition_misalignment.py \
    --input "${jsonl}" \
    --output-dir "${analysis_dir}" \
    --near-window 8 \
    --far-window 32 \
    --tau 8 \
    --no-figures \
    --skip-regression | tee "${ROOT}/analyze_${mode}.log"

  echo "=== video-cluster bootstrap: ${mode} ==="
  "${PYTHON_BIN}" tools/bootstrap_transition_misalignment.py \
    --input "${analysis_dir}/frame_with_transition_features.csv" \
    --output-dir "${bootstrap_dir}" \
    --near-window 8 \
    --far-window 32 \
    --bootstrap "${BOOTSTRAP}" \
    --seed "${SEED}" \
    --backend "${BOOTSTRAP_BACKEND}" | tee "${ROOT}/bootstrap_${mode}.log"

  if [[ "${KEEP_INTERMEDIATES}" != "1" ]]; then
    echo "=== cleanup heavy intermediates: mode=${mode} ==="
    rm -f "${jsonl}"
    rm -f "${analysis_dir}/frame_with_transition_features.csv"
    rm -f "${analysis_dir}/transition_curves.csv"
    rm -f "${analysis_dir}"/*.png
  fi

  echo "Analyzer directory: ${analysis_dir}"
  echo "ATM summary: ${analysis_dir}/atm_summary.csv"
  echo "Bootstrap summary: ${bootstrap_dir}/bootstrap_summary.csv"
  echo "Readable bootstrap summary: ${bootstrap_dir}/summary_readable.md"
}

run_mode "max_prob" "0.50 0.70 0.90"
run_mode "entropy" "1.0 2.0"
run_mode "top2_margin" "0.30 0.50 0.78"

echo "=== aggregate v1.7 summaries ==="
"${PYTHON_BIN}" - "${ROOT}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
modes = ["max_prob", "entropy", "top2_margin"]

atm_frames = []
bootstrap_frames = []
for mode in modes:
    atm_path = root / f"analysis_{mode}" / "atm_summary.csv"
    boot_path = root / f"bootstrap_{mode}" / "bootstrap_summary.csv"
    if atm_path.exists():
        atm = pd.read_csv(atm_path)
        atm.insert(0, "source_mode", mode)
        atm_frames.append(atm)
    if boot_path.exists():
        boot = pd.read_csv(boot_path)
        boot.insert(0, "source_mode", mode)
        bootstrap_frames.append(boot)

if atm_frames:
    pd.concat(atm_frames, ignore_index=True).to_csv(root / "atm_summary.csv", index=False)
if bootstrap_frames:
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(root / "bootstrap_summary.csv", index=False)
PY

echo "=== v1.7 causal TCN clean replication finished ==="
echo "Output root: ${ROOT}"
echo "Environment: ${ROOT}/environment.md"
echo "nvidia-smi: ${ROOT}/nvidia_smi.txt"
echo "Run config: ${ROOT}/run_config.md"
echo "Training/config log: ${ROOT}/training_log.md and ${ROOT}/run_*.log"
echo "Aggregate ATM summary: ${ROOT}/atm_summary.csv"
echo "Aggregate bootstrap summary: ${ROOT}/bootstrap_summary.csv"
echo "Per-mode analyzer summaries:"
echo "  ${ROOT}/analysis_max_prob/atm_summary.csv"
echo "  ${ROOT}/analysis_entropy/atm_summary.csv"
echo "  ${ROOT}/analysis_top2_margin/atm_summary.csv"
echo "Per-mode bootstrap summaries:"
echo "  ${ROOT}/bootstrap_max_prob/bootstrap_summary.csv"
echo "  ${ROOT}/bootstrap_entropy/bootstrap_summary.csv"
echo "  ${ROOT}/bootstrap_top2_margin/bootstrap_summary.csv"
echo "Reminder: package only lightweight summaries; do not commit JSONL, logs, caches, or generated artifacts."
