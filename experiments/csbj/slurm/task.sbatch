#!/bin/bash
# One task of E1: dataset $DS, task id = SLURM_ARRAY_TASK_ID (r = id // 5, f = id % 5),
# hyperparameter configuration $HP (default or regularized). Submitted by submit.sh.
#SBATCH --job-name=fedrf_e1
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/data/home/jorge.miguel.ferreira.silva/storage/fedrf_csbj/logs/%x_%A_%a.out
#SBATCH --error=/data/home/jorge.miguel.ferreira.silva/storage/fedrf_csbj/logs/%x_%A_%a.err
set -eo pipefail
source /data/home/jorge.miguel.ferreira.silva/storage/fedrf_csbj/code/experiments/csbj/slurm/env.sh
cd "$P/code"
echo "start $(date -Is) host $(hostname) job ${SLURM_JOB_ID} array ${SLURM_ARRAY_TASK_ID} ds ${DS} hp ${HP:-default} restarts ${SLURM_RESTART_COUNT:-0}"
python experiments/csbj/run_task.py --dataset "$DS" --task-id "$SLURM_ARRAY_TASK_ID" --hp "${HP:-default}"
echo "end $(date -Is)"
