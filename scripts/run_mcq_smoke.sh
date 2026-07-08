#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# One-command MCQ smoke-test entry point.
#
# Prerequisites — download from https://huggingface.co/datasets/facebook/S-EMBER:
#   * data/sember_mcq.jsonl
#   * data/videos/<video_id>.mp4         (one or more — pick any that interest you)
#
# This script then:
#   1. Installs the package (pip install -e ".[video,video-legacy]")
#   2. Filters data/sember_mcq.jsonl down to questions whose video is on disk
#      (writes the filtered JSONL in place; preserves the original as
#       data/sember_mcq.full.jsonl on first run).
#   3. Exports SEMBER_VIDEO_DIR and runs the chosen model.
#
# Usage:
#   bash scripts/run_mcq_smoke.sh                        # defaults: internvl3_5 4B
#   bash scripts/run_mcq_smoke.sh internvl3_5 8B
#   bash scripts/run_mcq_smoke.sh qwen3vl    4B

set -euo pipefail

MODEL="${1:-internvl3_5}"
SIZE="${2:-4B}"

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

DATA_JSONL="${REPO_ROOT}/data/sember_mcq.jsonl"
VIDEO_DIR="${REPO_ROOT}/data/videos"

# --- Pre-flight: data must already be on disk ---------------------------------
if [[ ! -f "${DATA_JSONL}" ]] || ! ls "${VIDEO_DIR}"/*.mp4 >/dev/null 2>&1; then
  cat >&2 <<EOF
ERROR: S-EMBER data not found.

Expected (any of these can be missing — the message tells you which):
  * ${DATA_JSONL}        $( [[ -f "${DATA_JSONL}" ]] && echo "[OK]" || echo "[MISSING]" )
  * ${VIDEO_DIR}/*.mp4   $( ls "${VIDEO_DIR}"/*.mp4 >/dev/null 2>&1 && echo "[OK]" || echo "[MISSING]" )

To fix:
  1. Install the Hugging Face CLI:
       pip install -U huggingface_hub
  2. Download:
       huggingface-cli download facebook/S-EMBER sember_mcq.jsonl --repo-type dataset --local-dir data
       huggingface-cli download facebook/S-EMBER --repo-type dataset --include 'videos/*.mp4' --local-dir data
     Or manually download a few mp4 files from:
       https://huggingface.co/datasets/facebook/S-EMBER/tree/main/videos
     and place them in ${VIDEO_DIR}/.
  3. Re-run:
       NUM_PROCESSES=1 bash scripts/run_mcq_smoke.sh ${MODEL} ${SIZE}

The runner will automatically restrict the evaluation to whichever
videos you downloaded — you do NOT need to grab all of them.
EOF
  exit 1
fi

echo "==[1/3] Installing the package (editable, with video extras)=="
pip install -e ".[video,video-legacy]" --quiet

echo "==[2/3] Filtering JSONL to videos on disk=="
python scripts/data/filter_jsonl_by_videos.py

export SEMBER_VIDEO_DIR="${VIDEO_DIR}"
echo "    SEMBER_VIDEO_DIR=${SEMBER_VIDEO_DIR}"

case "${MODEL}" in
  internvl3_5) RUNNER="scripts/mcq/run_internvl3_5.sh" ;;
  qwen3vl)     RUNNER="scripts/mcq/run_qwen3vl.sh"     ;;
  *) echo "ERROR: unknown model '${MODEL}'. Use 'internvl3_5' or 'qwen3vl'." >&2; exit 1 ;;
esac

echo "==[3/3] Running ${MODEL} ${SIZE}=="
exec bash "${RUNNER}" "${SIZE}"
