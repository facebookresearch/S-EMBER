#!/bin/bash
# SLURM submission wrapper for S-EMBER InternVL3.5 MCQ runs.
#
# Usage:
#   sbatch --export=ALL,SEMBER_VIDEO_DIR=/abs/path/to/videos,MODEL_SIZE=8B \
#          scripts/mcq/slurm/internvl3_5.sbatch
#
# The MODEL_SIZE variable selects the InternVL3.5 checkpoint. SEMBER_VIDEO_DIR
# is required by the inner runner. Other env vars accepted by
# scripts/mcq/run_internvl3_5.sh (NUM_FRAME, TOTAL_MAX_NUM, ...) can be
# forwarded the same way.
#
# Adjust the SBATCH directives below for your cluster.

#SBATCH --job-name=sember_mcq_internvl3_5
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
  echo "ERROR: MODEL_SIZE is not set (4B / 8B / 38B)." >&2
  exit 1
fi

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec "${REPO_ROOT}/scripts/mcq/run_internvl3_5.sh" "${MODEL_SIZE}"
