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

TRAIN_CSV="${TRAIN_CSV:-${REPO_ROOT}/data/tadA_lora_set.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/tadA_lora_new_model}"

mkdir -p "${OUTPUT_DIR}"

echo "[SPIN-JEvo] Training LoRA model"
echo "  base model: ${MODEL_NAME}"
echo "  train csv : ${TRAIN_CSV}"
echo "  save dir  : ${OUTPUT_DIR}"

python "${REPO_ROOT}/src/train_lora_classifier.py" \
  --model-name "${MODEL_NAME}" \
  --train-csv "${TRAIN_CSV}" \
  --output-dir "${OUTPUT_DIR}"
