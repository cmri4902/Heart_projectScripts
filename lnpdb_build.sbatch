#!/bin/bash
#SBATCH --job-name=HEART_LNPDB_3D
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=1-12:00:00
#SBATCH --output=HEART_LNPDB_%A_%a.out
#SBATCH --error=HEART_LNPDB_%A_%a.err
# Submit with: sbatch --export=ALL --array=0-15 lnpdb_build.sbatch
# Select your cluster's correct CPU partition and QoS at submission time.
set -euo pipefail
: "${ADAPTER:?Set ADAPTER}"
: "${REPO:?Set REPO}"
: "${LNPDB_PREPARED:?Set LNPDB_PREPARED}"
: "${GENERAL_DICT:?Set GENERAL_DICT}"
: "${LNPDB_FULL_OUT:?Set LNPDB_FULL_OUT}"
: "${PY:?Set PY to heart_gpu Python}"
: "${NUM_SHARDS:?Set NUM_SHARDS to the array length}"
: "${SLURM_ARRAY_TASK_ID:?Run as a Slurm array task}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" "$ADAPTER" build \
    --repo "$REPO" --prepared "$LNPDB_PREPARED" \
    --dict "$GENERAL_DICT" --out "$LNPDB_FULL_OUT" \
    --num-shards "$NUM_SHARDS" --shard-id "$SLURM_ARRAY_TASK_ID" \
    --max-atoms 256 --max-try 15 --min-3d 1 --min-distinct-3d 1
