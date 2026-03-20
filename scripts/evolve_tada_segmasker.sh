#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_LOCAL_MODEL="/mnt/hdd/chenzh/esm2_t33_650M_UR50D"
DEFAULT_HF_MODEL="facebook/esm2_t33_650M_UR50D"

MODEL_NAME="${MODEL_PATH:-${1:-}}"
if [[ -z "${MODEL_NAME}" ]]; then
  if [[ -d "${DEFAULT_LOCAL_MODEL}" ]]; then
    MODEL_NAME="${DEFAULT_LOCAL_MODEL}"
  else
    MODEL_NAME="${DEFAULT_HF_MODEL}"
  fi
fi

ADAPTER_PATH="${ADAPTER_PATH:-${REPO_ROOT}/checkpoints/tadA_10_model/}"
INPUT_CSV="${INPUT_CSV:-${REPO_ROOT}/data/tadA_seed.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
RUN_TAG="${RUN_TAG:-1}"

mkdir -p "${OUTPUT_DIR}"

echo "[SPIN-JEvo] Running GA sampling"
echo "  base model : ${MODEL_NAME}"
echo "  adapter    : ${ADAPTER_PATH}"
echo "  input csv  : ${INPUT_CSV}"
echo "  output dir : ${OUTPUT_DIR}"
echo "  run tag    : ${RUN_TAG}"

python "${REPO_ROOT}/src/evolve_with_ga.py" \
  --model-name "${MODEL_NAME}" \
  --adapter-path "${ADAPTER_PATH}" \
  --input-csv "${INPUT_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-tag "${RUN_TAG}"
