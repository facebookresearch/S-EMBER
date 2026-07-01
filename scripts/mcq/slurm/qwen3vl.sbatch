#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# SLURM submission wrapper for S-EMBER Qwen3-VL MCQ runs.
#
# Usage:
#   sbatch --export=ALL,SEMBER_VIDEO_DIR=/abs/path/to/videos,MODEL_SIZE=8B \
#          scripts/mcq/slurm/qwen3vl.sbatch
#
# The MODEL_SIZE variable selects the Qwen3-VL checkpoint (4B / 8B / 32B).
# SEMBER_VIDEO_DIR is required by the inner runner. Other env vars accepted by
# scripts/mcq/run_qwen3vl.sh (UNIFORM_NFRAMES, FIXED_FRAME_PIXELS, ...) can be
# forwarded the same way.
#
# Adjust the SBATCH directives below for your cluster.

#SBATCH --job-name=sember_mcq_qwen3vl
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=1-00:00:00
#SBATCH --output=output/mcq/slurm/%x_%j.out
#SBATCH --error=output/mcq/slurm/%x_%j.err

set -euo pipefail
mkdir -p output/mcq/slurm

if [[ -z "${MODEL_SIZE:-}" ]]; then
  echo "ERROR: MODEL_SIZE is not set (4B / 8B / 32B)." >&2
  exit 1
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${REPO_ROOT}/scripts/mcq/run_qwen3vl.sh" "${MODEL_SIZE}"
