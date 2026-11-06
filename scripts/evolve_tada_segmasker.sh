#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_HF_MODEL="facebook/esm2_t33_650M_UR50D"

MODEL_NAME="${MODEL_PATH:-${1:-${DEFAULT_HF_MODEL}}}"

: "${ADAPTER_PATH:?Set ADAPTER_PATH to your locally trained LoRA adapter directory}"
INPUT_CSV="${INPUT_CSV:-${REPO_ROOT}/data/tadA_seed.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/outputs}"
RUN_TAG="${RUN_TAG:-1}"
GENERATIONS="${GENERATIONS:-100}"
INITIAL_SUBSTITUTION_FRACTION="${INITIAL_SUBSTITUTION_FRACTION:-0.20}"
MUTATION_RATE="${MUTATION_RATE:-0.02}"
MATING_POOL_FRACTION="${MATING_POOL_FRACTION:-0.75}"
SEED="${SEED:-8893}"

mkdir -p "${OUTPUT_DIR}"

echo "[SPIN-JEvo] Running GA sampling"
echo "  base model : ${MODEL_NAME}"
echo "  adapter    : ${ADAPTER_PATH}"
echo "  input csv  : ${INPUT_CSV}"
echo "  output dir : ${OUTPUT_DIR}"
echo "  run tag    : ${RUN_TAG}"
echo "  generations: ${GENERATIONS}"
echo "  seed       : ${SEED}"

python "${REPO_ROOT}/src/evolve_with_ga.py" \
  --model-name "${MODEL_NAME}" \
  --adapter-path "${ADAPTER_PATH}" \
  --input-csv "${INPUT_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-tag "${RUN_TAG}" \
  --generations "${GENERATIONS}" \
  --initial-substitution-fraction "${INITIAL_SUBSTITUTION_FRACTION}" \
  --mutation-rate "${MUTATION_RATE}" \
  --mating-pool-fraction "${MATING_POOL_FRACTION}" \
  --seed "${SEED}"
