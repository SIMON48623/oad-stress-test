#!/usr/bin/env bash
set -euo pipefail

# v1.6 formal run script.
# Run manually on a CUDA machine:
#   export OUTPUT_ROOT=/path/to/result/storage
#   chmod +x scripts/run_v16_gpu_shift_robustness.sh
#   bash scripts/run_v16_gpu_shift_robustness.sh
#
# Summary-only package command after completion:
#   tar -czf v16_gpu_shift_robustness_summary_only.tar.gz \
#     "${ROOT}/environment.md" \
#     "${ROOT}/nvidia_smi.txt" \
#     "${ROOT}/run_config.md" \
#     "${ROOT}/atm_summary.csv" \
#     "${ROOT}/bootstrap_summary.csv" \
#     "${ROOT}"/analysis_*_*/atm_summary.csv \
#     "${ROOT}"/analysis_*_*/summary_readable.md \
#     "${ROOT}"/bootstrap_*_*/bootstrap_summary.csv \
#     "${ROOT}"/bootstrap_*_*/summary_readable.md

# Set OUTPUT_ROOT to a location with enough space for intermediate arrays.
OUTPUT_ROOT="${OUTPUT_ROOT:-${RESULTS_ROOT:-results}}"
ROOT="${ROOT:-${OUTPUT_ROOT%/}/v1.6_gpu_shift_robustness}"
CONFIG="${CONFIG:-configs/thumos14.yaml}"
FEATURE_DIR="${FEATURE_DIR:-data/thumos14/features}"
PYTHON_BIN="${PYTHON_BIN:-python}"
BOOTSTRAP="${BOOTSTRAP:-1000}"
BOOTSTRAP_BACKEND="${BOOTSTRAP_BACKEND:-numpy}"
SEED="${SEED:-42}"
GRU_MAX_EPOCHS="${GRU_MAX_EPOCHS:-20}"
GRU_HIDDEN_DIM="${GRU_HIDDEN_DIM:-128}"
GRU_CHUNK_LENGTH="${GRU_CHUNK_LENGTH:-512}"
KEEP_INTERMEDIATES="${KEEP_INTERMEDIATES:-0}"

# Required by torch deterministic CUDA algorithms in some environments.
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

mkdir -p "${ROOT}"

echo "=== v1.6 CUDA environment check ==="
"${PYTHON_BIN}" - <<'PY' | tee "${ROOT}/environment.md"
from __future__ import annotations

import platform
import sys

try:
    import torch
except Exception as exc:  # noqa: BLE001 - clear server-side setup failure.
    raise SystemExit(f"ERROR: PyTorch import failed: {exc}") from exc

print("# v1.6 GPU Shift Robustness Environment")
print()
print(f"- python: {sys.version.split()[0]}")
print(f"- platform: {platform.platform()}")
print(f"- torch.__version__: {torch.__version__}")
print(f"- torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"- torch.cuda.device_count(): {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"- torch.cuda.get_device_name(0): {torch.cuda.get_device_name(0)}")
else:
    raise SystemExit("ERROR: v1.6 formal run requires CUDA; refusing to fall back to CPU.")
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "=== nvidia-smi ==="
  nvidia-smi | tee "${ROOT}/nvidia_smi.txt"
else
  echo "nvidia-smi not found on PATH" | tee "${ROOT}/nvidia_smi.txt"
fi

cat > "${ROOT}/run_config.md" <<EOF
# v1.6 GPU Shift Robustness Run Config

- dataset: THUMOS14
- model: causal_gru
- device: cuda
- train split: clean THUMOS14 train split from configs/thumos14.yaml
- eval split: full 211-video THUMOS14 test split
- protocol choice: train once on clean features and evaluate clean plus shifted test features
- rationale: preserves the existing clean-train/shifted-test protocol and avoids changing the model or training target under shift
- hidden_dim: ${GRU_HIDDEN_DIM}
- max_epochs: ${GRU_MAX_EPOCHS}
- early_stopping: not supported by the current CausalGRUClassifier
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
- conditions:
  - clean: severity 0.0, eval_feature_dir from config
  - feature_noise: implemented by scripts/make_shift_set.py --kind noise --severity 0.1
  - temporal_subsample: implemented by scripts/make_shift_set.py --kind temporal_subsample --severity 0.5
- per_frame_logging: enabled for analyzer input
- keep_intermediates: ${KEEP_INTERMEDIATES}
- cublas_workspace_config: ${CUBLAS_WORKSPACE_CONFIG}

Generated shifted feature directories live under ${ROOT}/shift_features/ and should not be committed.
If keep_intermediates is 0, per-frame JSONL and frame-level analyzer CSV files are deleted after successful bootstrap.
EOF

prepare_shift() {
  local condition="$1"
  local kind="$2"
  local severity="$3"
  local dst="${ROOT}/shift_features/${condition}"
  mkdir -p "${dst}"
  echo "=== create shifted features: condition=${condition} kind=${kind} severity=${severity} ==="
  "${PYTHON_BIN}" scripts/make_shift_set.py \
    --src "${FEATURE_DIR}" \
    --dst "${dst}" \
    --kind "${kind}" \
    --severity "${severity}" \
    --seed "${SEED}" | tee "${ROOT}/make_shift_${condition}.log"
}

run_condition_mode() {
  local condition="$1"
  local severity="$2"
  local eval_feature_dir="$3"
  local mode="$4"
  local thresholds="$5"
  local is_clean="$6"
  local jsonl="${ROOT}/per_frame_causal_gru_thumos14_${condition}_${mode}.jsonl"
  local analysis_dir="${ROOT}/analysis_${condition}_${mode}"
  local bootstrap_dir="${ROOT}/bootstrap_${condition}_${mode}"
  local run_log="${ROOT}/run_${condition}_${mode}.log"

  echo "=== v1.6 full CUDA eval: condition=${condition} mode=${mode} thresholds=${thresholds} ==="
  local eval_args=()
  if [[ "${is_clean}" != "true" ]]; then
    eval_args+=(--eval-feature-dir "${eval_feature_dir}")
  fi

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
    --shift-name "${condition}" \
    --shift-severity "${severity}" \
    --is-clean "${is_clean}" \
    --overwrite \
    --progress-every 20 \
    "${eval_args[@]}" | tee "${run_log}"

  echo "=== validate per-frame JSONL: condition=${condition} mode=${mode} ==="
  "${PYTHON_BIN}" tools/validate_per_frame_log.py --input "${jsonl}" | tee "${ROOT}/validate_${condition}_${mode}.log"

  echo "=== transition misalignment analysis: condition=${condition} mode=${mode} ==="
  "${PYTHON_BIN}" tools/analyze_transition_misalignment.py \
    --input "${jsonl}" \
    --output-dir "${analysis_dir}" \
    --near-window 8 \
    --far-window 32 \
    --tau 8 \
    --no-figures \
    --skip-regression | tee "${ROOT}/analyze_${condition}_${mode}.log"

  echo "=== video-cluster bootstrap: condition=${condition} mode=${mode} ==="
  "${PYTHON_BIN}" tools/bootstrap_transition_misalignment.py \
    --input "${analysis_dir}/frame_with_transition_features.csv" \
    --output-dir "${bootstrap_dir}" \
    --near-window 8 \
    --far-window 32 \
    --bootstrap "${BOOTSTRAP}" \
    --seed "${SEED}" \
    --backend "${BOOTSTRAP_BACKEND}" | tee "${ROOT}/bootstrap_${condition}_${mode}.log"

  if [[ "${KEEP_INTERMEDIATES}" != "1" ]]; then
    echo "=== cleanup heavy intermediates: condition=${condition} mode=${mode} ==="
    rm -f "${jsonl}"
    rm -f "${analysis_dir}/frame_with_transition_features.csv"
    rm -f "${analysis_dir}/transition_curves.csv"
    rm -f "${analysis_dir}"/*.png
  fi
}

run_condition() {
  local condition="$1"
  local severity="$2"
  local eval_feature_dir="$3"
  local is_clean="$4"
  run_condition_mode "${condition}" "${severity}" "${eval_feature_dir}" "max_prob" "0.50 0.70 0.90" "${is_clean}"
  run_condition_mode "${condition}" "${severity}" "${eval_feature_dir}" "entropy" "1.0 2.0" "${is_clean}"
  run_condition_mode "${condition}" "${severity}" "${eval_feature_dir}" "top2_margin" "0.30 0.50 0.78" "${is_clean}"
}

prepare_shift "feature_noise_s0p1" "noise" "0.1"
prepare_shift "temporal_subsample_s0p5" "temporal_subsample" "0.5"

run_condition "clean" "0.0" "" "true"
run_condition "feature_noise_s0p1" "0.1" "${ROOT}/shift_features/feature_noise_s0p1" "false"
run_condition "temporal_subsample_s0p5" "0.5" "${ROOT}/shift_features/temporal_subsample_s0p5" "false"

echo "=== aggregate v1.6 summaries ==="
"${PYTHON_BIN}" - "${ROOT}" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

root = Path(sys.argv[1])
conditions = ["clean", "feature_noise_s0p1", "temporal_subsample_s0p5"]
modes = ["max_prob", "entropy", "top2_margin"]

atm_frames = []
bootstrap_frames = []
for condition in conditions:
    for mode in modes:
        atm_path = root / f"analysis_{condition}_{mode}" / "atm_summary.csv"
        boot_path = root / f"bootstrap_{condition}_{mode}" / "bootstrap_summary.csv"
        if atm_path.exists():
            atm = pd.read_csv(atm_path)
            atm.insert(0, "source_condition", condition)
            atm.insert(1, "source_mode", mode)
            atm_frames.append(atm)
        if boot_path.exists():
            boot = pd.read_csv(boot_path)
            boot.insert(0, "source_condition", condition)
            boot.insert(1, "source_mode", mode)
            bootstrap_frames.append(boot)

if atm_frames:
    pd.concat(atm_frames, ignore_index=True).to_csv(root / "atm_summary.csv", index=False)
if bootstrap_frames:
    pd.concat(bootstrap_frames, ignore_index=True).to_csv(root / "bootstrap_summary.csv", index=False)
PY

echo "=== v1.6 GPU shift robustness finished ==="
echo "Output root: ${ROOT}"
echo "Environment: ${ROOT}/environment.md"
echo "nvidia-smi: ${ROOT}/nvidia_smi.txt"
echo "Run config: ${ROOT}/run_config.md"
echo "Aggregate ATM summary: ${ROOT}/atm_summary.csv"
echo "Aggregate bootstrap summary: ${ROOT}/bootstrap_summary.csv"
echo "Shift feature dirs:"
echo "  ${ROOT}/shift_features/feature_noise_s0p1"
echo "  ${ROOT}/shift_features/temporal_subsample_s0p5"
echo "Reminder: package only lightweight summaries; do not commit JSONL, logs, caches, or generated feature files."
