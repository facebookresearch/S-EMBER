#!/bin/bash
# Run S-EMBER MCQ evaluation for a Qwen3-VL model (4B / 8B / 32B).
#
# Frame sampling defaults to the "uniform_768f_fixres" recipe used in the
# paper: 768 uniformly-sampled frames per video, each frame resized to a
# fixed 786,432-pixel budget (independent of source resolution).
#
# Usage:
#   scripts/mcq/run_qwen3vl.sh <MODEL_SIZE>
#
# Example:
#   scripts/mcq/run_qwen3vl.sh 8B
#
# Required environment variables:
#   SEMBER_VIDEO_DIR     Absolute path to the directory holding the benchmark mp4 files.
#                        Each sample's video is loaded from $SEMBER_VIDEO_DIR/<video_id>.mp4
#
# Optional environment variables:
#   SEMBER_OUTPUT_DIR    Where to write per-sample logs and the metrics json.
#                        Defaults to ./output/mcq/qwen3vl_<size>_uniform_<N>f_fixres
#   NUM_PROCESSES        Number of GPUs to launch with accelerate (default 8).
#   UNIFORM_NFRAMES      Frames sampled uniformly per video (default 768).
#   FIXED_FRAME_PIXELS   Per-frame pixel budget (default 786432).
#   MAIN_PROCESS_PORT    accelerate main process port (default 12515).
#
# Prerequisites:
#   * A jsonl benchmark file at  ./data/sember_mcq.jsonl
#   * pip install -e ".[video,video-legacy]" from the repo root.
#   * flash-attn installed (the runner sets attn_implementation=flash_attention_2).

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <MODEL_SIZE>   (one of: 4B 8B 32B)" >&2
  exit 1
fi

MODEL_SIZE="$1"
case "${MODEL_SIZE}" in
  4B|8B|32B) ;;
  *) echo "Unsupported MODEL_SIZE='${MODEL_SIZE}'. Choose 4B, 8B or 32B." >&2; exit 1 ;;
esac

if [[ -z "${SEMBER_VIDEO_DIR:-}" ]]; then
  echo "ERROR: SEMBER_VIDEO_DIR is not set." >&2
  echo "       Export it to the directory containing the benchmark .mp4 files." >&2
  exit 1
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"

UNIFORM_NFRAMES="${UNIFORM_NFRAMES:-768}"
FIXED_FRAME_PIXELS="${FIXED_FRAME_PIXELS:-786432}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-12515}"

SCRIPT_NAME="qwen3vl_${MODEL_SIZE,,}_uniform_${UNIFORM_NFRAMES}f_fixres"
PRETRAINED="Qwen/Qwen3-VL-${MODEL_SIZE}-Instruct"
OUTPUT_DIR="${SEMBER_OUTPUT_DIR:-${REPO_ROOT}/output/mcq/${SCRIPT_NAME}}"

mkdir -p "${OUTPUT_DIR}"

echo "=== S-EMBER Qwen3-VL MCQ run ==="
echo "  Model:               ${PRETRAINED}"
echo "  uniform_nframes:     ${UNIFORM_NFRAMES}"
echo "  fixed_frame_pixels:  ${FIXED_FRAME_PIXELS}"
echo "  GPUs:                ${NUM_PROCESSES}"
echo "  Video dir:           ${SEMBER_VIDEO_DIR}"
echo "  Output dir:          ${OUTPUT_DIR}"
echo "================================"

cd "${REPO_ROOT}"

accelerate launch \
    --num_processes="${NUM_PROCESSES}" \
    --main_process_port="${MAIN_PROCESS_PORT}" \
    -m lmms_eval eval \
    --model qwen3_vl \
    --model_args="pretrained=${PRETRAINED},attn_implementation=flash_attention_2,interleave_visuals=False,uniform_nframes=${UNIFORM_NFRAMES},fixed_frame_pixels=${FIXED_FRAME_PIXELS}" \
    --tasks sember_mcq \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix "${SCRIPT_NAME}_mcq" \
    --output_path "${OUTPUT_DIR}" \
    --force_simple \
    --verbosity DEBUG
