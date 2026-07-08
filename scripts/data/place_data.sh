#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Move downloaded S-EMBER files into the canonical local data layout.
#
# Usage:
#   bash scripts/data/place_data.sh <download-dir>
#
# Example:
#   bash scripts/data/place_data.sh ~/Downloads
#
# Effect:
#   <download-dir>/sember_mcq*.jsonl        ->  data/sember_mcq.jsonl
#   <download-dir>/sember_grounding*.jsonl  ->  data/sember_grounding.jsonl
#   <download-dir>/*.mp4                    ->  data/videos/
#
# The script never deletes anything that didn't match. Pre-existing files
# in the destination are overwritten.

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: bash scripts/data/place_data.sh <download-dir>" >&2
  exit 1
fi

SRC_DIR="$1"
if [[ ! -d "${SRC_DIR}" ]]; then
  echo "ERROR: ${SRC_DIR} is not a directory" >&2
  exit 1
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${REPO_ROOT}/data"
VIDEO_DIR="${DATA_DIR}/videos"
mkdir -p "${VIDEO_DIR}"

MOVED_JSONL=0
MOVED_MP4=0

shopt -s nullglob
for f in "${SRC_DIR}"/sember_mcq*.jsonl; do
  echo "  jsonl: $(basename "$f") -> ${DATA_DIR}/sember_mcq.jsonl"
  cp -f "$f" "${DATA_DIR}/sember_mcq.jsonl"
  MOVED_JSONL=$((MOVED_JSONL + 1))
done

for f in "${SRC_DIR}"/sember_grounding*.jsonl; do
  echo "  jsonl: $(basename "$f") -> ${DATA_DIR}/sember_grounding.jsonl"
  cp -f "$f" "${DATA_DIR}/sember_grounding.jsonl"
  MOVED_JSONL=$((MOVED_JSONL + 1))
done

for f in "${SRC_DIR}"/*.mp4; do
  echo "  mp4:   $(basename "$f") -> ${VIDEO_DIR}/"
  cp -f "$f" "${VIDEO_DIR}/$(basename "$f")"
  MOVED_MP4=$((MOVED_MP4 + 1))
done
shopt -u nullglob

echo
echo "Placed: ${MOVED_JSONL} jsonl + ${MOVED_MP4} mp4(s)."
if (( MOVED_JSONL == 0 && MOVED_MP4 == 0 )); then
  echo "WARNING: no matching files found in ${SRC_DIR}." >&2
  echo "         Expected sember_mcq*.jsonl, sember_grounding*.jsonl, and/or *.mp4." >&2
  exit 1
fi

echo
echo "Next:"
echo "  bash scripts/run_mcq_smoke.sh internvl3_5 4B"
