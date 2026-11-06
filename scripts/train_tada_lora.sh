#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_HF_MODEL="facebook/esm2_t33_650M_UR50D"

MODEL_NAME="${MODEL_PATH:-${1:-${DEFAULT_HF_MODEL}}}"

TRAIN_CSV="${TRAIN_CSV:-${REPO_ROOT}/data/training/tada_round1_train_historical.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/tadA_lora_new_model}"
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
SEED="${SEED:-8893}"

mkdir -p "${OUTPUT_DIR}"

echo "[SPIN-JEvo] Training LoRA model"
echo "  base model: ${MODEL_NAME}"
echo "  train csv : ${TRAIN_CSV}"
echo "  save dir  : ${OUTPUT_DIR}"
echo "  epochs    : ${EPOCHS}"
echo "  batch size: ${BATCH_SIZE}"
echo "  seed      : ${SEED}"

python "${REPO_ROOT}/src/train_lora_classifier.py" \
  --model-name "${MODEL_NAME}" \
  --train-csv "${TRAIN_CSV}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --seed "${SEED}" \
  --max-grad-norm 0.5
