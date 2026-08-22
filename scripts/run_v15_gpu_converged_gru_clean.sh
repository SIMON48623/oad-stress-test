#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-results/v1.5_gpu_converged_gru_clean}"
CONFIG="${CONFIG:-configs/thumos14.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BOOTSTRAP="${BOOTSTRAP:-1000}"
SEED="${SEED:-42}"
GRU_MAX_EPOCHS="${GRU_MAX_EPOCHS:-20}"
GRU_HIDDEN_DIM="${GRU_HIDDEN_DIM:-128}"
GRU_CHUNK_LENGTH="${GRU_CHUNK_LENGTH:-512}"

mkdir -p "${ROOT}"

echo "=== v1.5 CUDA environment check ==="
"${PYTHON_BIN}" - <<'PY' | tee "${ROOT}/environment.md"
from __future__ import annotations

import platform
import sys

try:
    import torch
except Exception as exc:  # noqa: BLE001 - clear server-side setup failure.
    raise SystemExit(f"ERROR: PyTorch import failed: {exc}") from exc

print("# v1.5 GPU Environment")
print()
print(f"- python: {sys.version.split()[0]}")
print(f"- platform: {platform.platform()}")
print(f"- torch.__version__: {torch.__version__}")
print(f"- torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"- torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"- torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("ERROR: v1.5 formal run requires CUDA; refusing to fall back to CPU.")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== nvidia-smi ==="
  nvidia-smi | tee "${ROOT}/nvidia_smi.txt"
else
  echo "nvidia-smi not found on PATH" | tee "${ROOT}/nvidia_smi.txt"
fi

cat > "${ROOT}/run_config.md" <<EOF
# v1.5 Converged GRU Clean Run Config

- dataset: THUMOS14 clean
- model: causal_gru
- device: cuda
- test split: full 211-video THUMOS14 test split
- hidden_dim: ${GRU_HIDDEN_DIM}
- max_epochs: ${GRU_MAX_EPOCHS}
- early_stopping: not supported by the current CausalGRUClassifier
- patience: not applied
- bootstrap: ${BOOTSTRAP}
- bootstrap_unit: video
- near_window: 8
- far_window: 32
- policy: confidence_threshold
- budgets: 0.25, 1.00
- max_videos: none
- per_frame_logging: enabled

The training diagnostics are written by scripts/run_baseline.py to stdout and
training_log.md. The first mode should fit the GRU; later modes should reuse the
same cache key for the converged CUDA classifier.
EOF

run_mode() {
  local mode="$1"
  local thresholds="$2"
  local jsonl="${ROOT}/per_frame_causal_gru_thumos14_clean_${mode}.jsonl"
  local analysis_dir="${ROOT}/analysis_${mode}"
  local bootstrap_dir="${ROOT}/bootstrap_${mode}"
  local run_log="${ROOT}/run_${mode}.log"

  echo "=== v1.5 full CUDA eval: mode=${mode} thresholds=${thresholds} ==="
  # shellcheck disable=SC2086
  "${PYTHON_BIN}" scripts/run_baseline.py \
    --config "${CONFIG}" \
    --classifier causal_gru \
    --policy confidence_threshold \
    --budgets 0.25 1.00 \
    --thresholds ${thresholds} \
    --uncertainty-mode "${mode}" \
    --summary-only \
    --dump-per-frame \
    --per-frame-output "${jsonl}" \
    --results-dir "${ROOT}" \
    --gru-device cuda \
    --gru-hidden-dim "${GRU_HIDDEN_DIM}" \
    --gru-max-epochs "${GRU_MAX_EPOCHS}" \
    --gru-chunk-length "${GRU_CHUNK_LENGTH}" \
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
    --seed "${SEED}" | tee "${ROOT}/bootstrap_${mode}.log"

  echo "JSONL: ${jsonl}"
  echo "Analyzer directory: ${analysis_dir}"
  echo "ATM summary: ${analysis_dir}/atm_summary.csv"
  echo "Bootstrap summary: ${bootstrap_dir}/bootstrap_summary.csv"
  echo "Readable bootstrap summary: ${bootstrap_dir}/summary_readable.md"
}

run_mode "max_prob" "0.50 0.70 0.90"
run_mode "entropy" "1.0 2.0"
run_mode "top2_margin" "0.30 0.50 0.78"

echo "=== aggregate v1.5 summaries ==="
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

echo "=== v1.5 converged GRU clean replication finished ==="
echo "Output root: ${ROOT}"
echo "Environment: ${ROOT}/environment.md"
echo "nvidia-smi: ${ROOT}/nvidia_smi.txt"
echo "Training/config log: ${ROOT}/training_log.md and ${ROOT}/run_*.log"
echo "Aggregate ATM summary: ${ROOT}/atm_summary.csv"
echo "Aggregate bootstrap summary: ${ROOT}/bootstrap_summary.csv"
echo "Per-mode bootstrap summaries:"
echo "  ${ROOT}/bootstrap_max_prob/bootstrap_summary.csv"
echo "  ${ROOT}/bootstrap_entropy/bootstrap_summary.csv"
echo "  ${ROOT}/bootstrap_top2_margin/bootstrap_summary.csv"
