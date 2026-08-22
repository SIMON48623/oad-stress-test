#!/usr/bin/env bash
set -euo pipefail

ROOT="results/v1.1_gru_clean_go_nogo/full"
CONFIG="configs/thumos14.yaml"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${ROOT}"

run_mode() {
  local mode="$1"
  local thresholds="$2"
  local jsonl="${ROOT}/per_frame_causal_gru_thumos14_clean_${mode}.jsonl"
  local analysis_dir="${ROOT}/analysis_${mode}"

  echo "=== v1.1 full GPU eval: ${mode} thresholds=${thresholds} ==="
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
    --gru-max-epochs 1 \
    --gru-chunk-length 512 \
    --shift-name clean \
    --overwrite \
    --progress-every 20

  echo "=== validate per-frame JSONL: ${mode} ==="
  "${PYTHON_BIN}" tools/validate_per_frame_log.py --input "${jsonl}"

  echo "=== transition misalignment analysis: ${mode} ==="
  "${PYTHON_BIN}" tools/analyze_transition_misalignment.py \
    --input "${jsonl}" \
    --output-dir "${analysis_dir}" \
    --near-window 8 \
    --far-window 32 \
    --tau 8

  echo "JSONL: ${jsonl}"
  echo "Analyzer directory: ${analysis_dir}"
  echo "ATM summary: ${analysis_dir}/atm_summary.csv"
  echo "Readable summary: ${analysis_dir}/summary_readable.md"
}

run_mode "max_prob" "0.50 0.70 0.90"
run_mode "entropy" "1.0 2.0"
run_mode "top2_margin" "0.30 0.50 0.78"

echo "=== v1.1 full GPU clean go/no-go finished ==="
echo "Per-frame JSONLs:"
echo "  ${ROOT}/per_frame_causal_gru_thumos14_clean_max_prob.jsonl"
echo "  ${ROOT}/per_frame_causal_gru_thumos14_clean_entropy.jsonl"
echo "  ${ROOT}/per_frame_causal_gru_thumos14_clean_top2_margin.jsonl"
echo "Analyzer directories:"
echo "  ${ROOT}/analysis_max_prob"
echo "  ${ROOT}/analysis_entropy"
echo "  ${ROOT}/analysis_top2_margin"
echo "ATM summaries:"
echo "  ${ROOT}/analysis_max_prob/atm_summary.csv"
echo "  ${ROOT}/analysis_entropy/atm_summary.csv"
echo "  ${ROOT}/analysis_top2_margin/atm_summary.csv"
echo "Readable summaries:"
echo "  ${ROOT}/analysis_max_prob/summary_readable.md"
echo "  ${ROOT}/analysis_entropy/summary_readable.md"
echo "  ${ROOT}/analysis_top2_margin/summary_readable.md"
