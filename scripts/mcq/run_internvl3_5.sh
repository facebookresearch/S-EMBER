#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Run S-EMBER MCQ evaluation for an InternVL3.5 model (4B / 8B / 38B).
#
# Usage:
#   scripts/mcq/run_internvl3_5.sh <MODEL_SIZE>
#
# Example:
#   scripts/mcq/run_internvl3_5.sh 8B
#
# Required environment variables:
#   SEMBER_VIDEO_DIR   Absolute path to the directory holding the benchmark mp4 files.
#                      Each sample's video is loaded from $SEMBER_VIDEO_DIR/<video_id>.mp4
#
# Optional environment variables:
#   SEMBER_OUTPUT_DIR  Where to write per-sample logs and the metrics json.
#                      Defaults to ./output/mcq/internvl3_5_<size>
#   NUM_PROCESSES      Number of GPUs to launch with accelerate (default 8).
#   NUM_FRAME          Frames sampled per video (default 128).
#   TOTAL_MAX_NUM      Upper bound on total image tiles (default 128).
#   MAIN_PROCESS_PORT  accelerate main process port (default 12530).
#
# Prerequisites:
#   * A jsonl benchmark file at  ./data/sember_mcq.jsonl
#   * pip install -e ".[video,video-legacy]" from the repo root.

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <MODEL_SIZE>   (one of: 4B 8B 38B)" >&2
  exit 1
fi

MODEL_SIZE="$1"
case "${MODEL_SIZE}" in
  4B|8B|38B) ;;
  *) echo "Unsupported MODEL_SIZE='${MODEL_SIZE}'. Choose 4B, 8B or 38B." >&2; exit 1 ;;
esac

if [[ -z "${SEMBER_VIDEO_DIR:-}" ]]; then
  echo "ERROR: SEMBER_VIDEO_DIR is not set." >&2
  echo "       Export it to the directory containing the benchmark .mp4 files." >&2
  exit 1
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

SCRIPT_NAME="internvl3_5_${MODEL_SIZE,,}"
PRETRAINED="OpenGVLab/InternVL3_5-${MODEL_SIZE}"

NUM_PROCESSES="${NUM_PROCESSES:-8}"
NUM_FRAME="${NUM_FRAME:-128}"
TOTAL_MAX_NUM="${TOTAL_MAX_NUM:-128}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12530}"
OUTPUT_DIR="${SEMBER_OUTPUT_DIR:-${REPO_ROOT}/output/mcq/${SCRIPT_NAME}}"

mkdir -p "${OUTPUT_DIR}"

echo "=== S-EMBER InternVL3.5 MCQ run ==="
echo "  Model:         ${PRETRAINED}"
echo "  Frames:        ${NUM_FRAME}"
echo "  total_max_num: ${TOTAL_MAX_NUM}"
echo "  GPUs:          ${NUM_PROCESSES}"
echo "  Video dir:     ${SEMBER_VIDEO_DIR}"
echo "  Output dir:    ${OUTPUT_DIR}"
echo "==================================="

cd "${REPO_ROOT}"

# Pre-download the model weights on a single process to dodge a multi-process
# race when several ranks try to populate the HF cache simultaneously.
python -c "
from transformers import AutoModel, AutoTokenizer
AutoModel.from_pretrained('${PRETRAINED}', trust_remote_code=True, device_map='cpu', torch_dtype='auto')
AutoTokenizer.from_pretrained('${PRETRAINED}', trust_remote_code=True, use_fast=False)
print('Weights cached for ${PRETRAINED}.')
"

accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --main_process_port="${MAIN_PROCESS_PORT}" \
    -m lmms_eval eval \
    --model internvl3_5 \
    --model_args "pretrained=${PRETRAINED},modality=video,num_frame=${NUM_FRAME},total_max_num=${TOTAL_MAX_NUM}" \
    --tasks sember_mcq \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${SCRIPT_NAME}_mcq" \
    --output_path "${OUTPUT_DIR}" \
    --verbosity DEBUG
