#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SLURM submission wrapper for S-EMBER InternVL3.5 grounding runs.
#
# Usage:
#   sbatch --export=ALL,SEMBER_VIDEO_DIR=/abs/path/to/videos,MODEL_SIZE=8B \
#          scripts/grounding/slurm/internvl3_5.sbatch
#
# MODEL_SIZE is required (4B / 8B / 38B). Other env vars accepted by
# scripts/grounding/run_internvl3_5.sh (NUM_FRAME, TOTAL_MAX_NUM, ...)
# can be forwarded the same way.

#SBATCH --job-name=sember_grnd_internvl3_5
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=output/grounding/slurm/%x_%j.out
#SBATCH --error=output/grounding/slurm/%x_%j.err

set -euo pipefail
mkdir -p output/grounding/slurm

if [[ -z "${MODEL_SIZE:-}" ]]; then
  echo "ERROR: MODEL_SIZE is not set (4B / 8B / 38B)." >&2
  exit 1
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${REPO_ROOT}/scripts/grounding/run_internvl3_5.sh" "${MODEL_SIZE}"
