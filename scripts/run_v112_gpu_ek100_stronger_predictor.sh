#!/usr/bin/env bash
set -euo pipefail

# Formal v1.12 EK100 stronger-predictor confound check.
# This is a diagnostic reference run, not a new method or SOTA claim.
#
# Required:
#   export EK100_ROOT=/path/to/extracted
# Optional:
#   export OUTPUT_ROOT=/path/to/large/result/storage
# Then:
#   bash scripts/run_v112_gpu_ek100_stronger_predictor.sh

PYTHON_BIN="${PYTHON_BIN:-python}"
EK100_ROOT="${EK100_ROOT:?Set EK100_ROOT to the extracted TeSTra EK100 root}"
OUTPUT_ROOT="${OUTPUT_ROOT:-results}"
ROOT="${OUTPUT_ROOT%/}/v1.12_ek100_stronger_predictor_sanity"
DOC_ROOT="${ROOT}/reports"
CANONICAL="${ROOT}/per_frame_causal_tcn_ek100_clean.jsonl"
SEED="${SEED:-20260607}"

mkdir -p "${ROOT}" "${DOC_ROOT}"

"${PYTHON_BIN}" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("cuda_device_count:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("ERROR: formal v1.12 requires CUDA; refusing CPU fallback.")
print("cuda_device:", torch.cuda.get_device_name(0))
PY

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee "${ROOT}/nvidia_smi.txt"
fi

"${PYTHON_BIN}" scripts/run_v112_ek100_stronger_predictor.py \
  --formal \
  --ek100-root "${EK100_ROOT}" \
  --output-dir "${ROOT}" \
  --quality-doc "${DOC_ROOT}/predictor_quality.md" \
  --max-train-sessions 500 \
  --max-eval-sessions 133 \
  --max-frames-per-session 1000000 \
  --hidden-dim 128 \
  --num-layers 4 \
  --kernel-size 3 \
  --dilations 1 2 4 8 \
  --dropout 0.0 \
  --epochs 20 \
  --lr 0.001 \
  --chunk-length 512 \
  --device cuda \
  --seed "${SEED}"

"${PYTHON_BIN}" tools/validate_per_frame_log.py --input "${CANONICAL}"

"${PYTHON_BIN}" - "${ROOT}/predictor_quality.csv" <<'PY'
import csv
import sys
with open(sys.argv[1], newline="", encoding="utf-8") as handle:
    row = next(csv.DictReader(handle))
if row["materially_improved"].lower() != "true":
    raise SystemExit(
        "QUALITY GATE STOP: predictor raw error did not improve by the "
        "pre-specified 0.02 absolute margin; ladder was not run."
    )
print("quality gate passed; proceeding to reliability ladder")
PY

DRY="${ROOT}/ladder_dryrun"
mkdir -p "${DRY}"
"${PYTHON_BIN}" tools/reliability_ladder.py \
  --input "${CANONICAL}" \
  --dataset ek100 \
  --model causal_tcn \
  --out "${DRY}/details" \
  --summary-csv "${DRY}/ek100_ladder_primary_c080.csv" \
  --summary-md "${DRY}/ek100_ladder_primary_c080.md" \
  --curve-csv "${DRY}/ek100_risk_coverage_curve.csv" \
  --near 8 --far 32 --coverage 0.80 \
  --score one_minus_max_prob --calib-frac 0.50 \
  --n-boot 100 --seed "${SEED}" --expected-videos 133 \
  --curve-coverages 0.70 0.80 0.90

"${PYTHON_BIN}" tools/reliability_ladder.py \
  --input "${CANONICAL}" \
  --dataset ek100 \
  --model causal_tcn \
  --out "${ROOT}/ladder_details" \
  --summary-csv "${DOC_ROOT}/ek100_ladder_primary_c080.csv" \
  --summary-md "${DOC_ROOT}/ek100_ladder_primary_c080.md" \
  --curve-csv "${DOC_ROOT}/ek100_risk_coverage_curve.csv" \
  --near 8 --far 32 --coverage 0.80 \
  --score one_minus_max_prob --calib-frac 0.50 \
  --n-boot 2000 --seed "${SEED}" --expected-videos 133 \
  --curve-coverages 0.70 0.80 0.90

"${PYTHON_BIN}" - \
  "${ROOT}/predictor_quality.csv" \
  "${DOC_ROOT}/ek100_ladder_primary_c080.csv" \
  "${DOC_ROOT}/v112_ek100_confound_check.md" <<'PY'
import csv
import sys
from pathlib import Path

quality_path, ladder_path, output_path = map(Path, sys.argv[1:])
with quality_path.open(newline="", encoding="utf-8") as handle:
    quality = next(csv.DictReader(handle))
with ladder_path.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
global_row = next(row for row in rows if row["Method"] == "Global split-conformal")
teg = float(global_row["TEG"])
teg_low = float(global_row["TEG CI low"])
teg_high = float(global_row["TEG CI high"])
error_or = float(global_row["Error_OR"])
or_low = float(global_row["Error_OR CI low"])
or_high = float(global_row["Error_OR CI high"])
transition_risk = float(global_row["Transition Risk"])
stable_risk = float(global_row["Stable Risk"])
improved = quality["materially_improved"].lower() == "true"

if improved and transition_risk > stable_risk and or_low > 1:
    conclusion = (
        "The earlier EK100 null was likely confounded by predictor weakness; "
        "cross-dataset structure claims must be weakened."
    )
elif improved and teg_low <= 0 <= teg_high and or_low <= 1 <= or_high:
    conclusion = (
        "The EK100 null is less likely to be solely a weak-predictor artifact, "
        "although it remains model- and protocol-dependent."
    )
else:
    conclusion = (
        "The stronger-predictor check is mixed or inconclusive; it does not "
        "support a stronger cross-dataset claim."
    )

output_path.write_text(
    "\n".join(
        [
            "# v1.12 EK100 Stronger-Predictor Confound Check",
            "",
            "This is a diagnostic reference run, not a new method or SOTA claim.",
            "",
            f"- predictor: causal_tcn",
            f"- raw_error_rate: {float(quality['raw_error_rate']):.6f}",
            f"- v1.10/v1.11 raw_error_rate: {float(quality['v110_raw_error_rate']):.6f}",
            f"- raw_error improvement: {float(quality['raw_error_improvement_vs_v110']):.6f}",
            f"- Global TEG: {teg:.6f} [{teg_low:.6f}, {teg_high:.6f}]",
            f"- Global Error_OR: {error_or:.6f} [{or_low:.6f}, {or_high:.6f}]",
            f"- transition risk / stable risk: {transition_risk:.6f} / {stable_risk:.6f}",
            "",
            f"Interpretation: {conclusion}",
            "",
            "The result remains specific to this reference model, EK100 active-label-set "
            "transition convention, scalar uncertainty score, and ladder protocol.",
            "",
        ]
    ),
    encoding="utf-8",
)
PY

echo "v1.12 complete"
echo "canonical JSONL (do not commit): ${CANONICAL}"
echo "quality: ${DOC_ROOT}/predictor_quality.md"
echo "ladder: ${DOC_ROOT}/ek100_ladder_primary_c080.csv"
echo "report: ${DOC_ROOT}/v112_ek100_confound_check.md"
