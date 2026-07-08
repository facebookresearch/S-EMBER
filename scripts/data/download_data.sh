#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Print step-by-step instructions for fetching the public S-EMBER dataset
# from Hugging Face.
#
# Usage:
#   bash scripts/data/download_data.sh
#
# Optional helper to move already-downloaded files:
#   bash scripts/data/place_data.sh ~/Downloads

set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
VIDEO_DIR="${DATA_DIR}/videos"
mkdir -p "${VIDEO_DIR}"

cat <<EOF
================================================================
S-EMBER dataset — Hugging Face download instructions
================================================================

Dataset repo:
  https://huggingface.co/datasets/facebook/S-EMBER/tree/main

Expected local layout:
  ${DATA_DIR}/sember_mcq.jsonl
  ${DATA_DIR}/sember_grounding.jsonl
  ${VIDEO_DIR}/<video_id>.mp4

Step 1.  Install the Hugging Face CLI:

             pip install -U huggingface_hub

Step 2.  Download the finalized JSONL files:

             huggingface-cli download facebook/S-EMBER sember_mcq.jsonl \
               --repo-type dataset --local-dir ${DATA_DIR}
             huggingface-cli download facebook/S-EMBER sember_grounding.jsonl \
               --repo-type dataset --local-dir ${DATA_DIR}

Step 3.  Download videos.

         Full dataset:

             huggingface-cli download facebook/S-EMBER --repo-type dataset \
               --include 'videos/*.mp4' --local-dir ${DATA_DIR}

         Small smoke test:
           Download a few mp4 files from
           https://huggingface.co/datasets/facebook/S-EMBER/tree/main/videos
           and place them in ${VIDEO_DIR}/ with their original filenames.

Step 4.  Run an MCQ smoke test:

             bash scripts/run_mcq_smoke.sh internvl3_5 4B

         The runner will:
           * filter sember_mcq.jsonl down to the questions whose video
             is on disk,
           * export SEMBER_VIDEO_DIR=${VIDEO_DIR},
           * launch the chosen model.

----------------------------------------------------------------
Convenience helper — if you saved files to ~/Downloads, run:
    bash scripts/data/place_data.sh ~/Downloads

It moves sember_mcq*.jsonl and sember_grounding*.jsonl into ${DATA_DIR}/
and any *.mp4 into ${VIDEO_DIR}/.
================================================================
EOF

# Show what is currently present so the user knows what's missing.
echo
echo "Currently on disk:"
[[ -f "${DATA_DIR}/sember_mcq.jsonl" ]] \
  && echo "  [OK]      ${DATA_DIR}/sember_mcq.jsonl ($(wc -l < "${DATA_DIR}/sember_mcq.jsonl") lines)" \
  || echo "  [MISSING] ${DATA_DIR}/sember_mcq.jsonl"
[[ -f "${DATA_DIR}/sember_grounding.jsonl" ]] \
  && echo "  [OK]      ${DATA_DIR}/sember_grounding.jsonl ($(wc -l < "${DATA_DIR}/sember_grounding.jsonl") lines)" \
  || echo "  [MISSING] ${DATA_DIR}/sember_grounding.jsonl"
N_VID=$(ls "${VIDEO_DIR}"/*.mp4 2>/dev/null | wc -l)
if [[ "${N_VID}" -gt 0 ]]; then
  echo "  [OK]      ${N_VID} mp4 file(s) in ${VIDEO_DIR}/"
else
  echo "  [MISSING] no mp4 files in ${VIDEO_DIR}/"
fi
