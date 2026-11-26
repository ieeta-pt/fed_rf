#!/bin/bash
#SBATCH --job-name=fedrf_report
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=01:30:00
#SBATCH --output=/data/home/jorge.miguel.ferreira.silva/storage/fedrf_csbj/logs/%x_%j.out
# Tables, numbers and figures. DATASETS is passed with --export, e.g.
#   sbatch --nice=100 --export=ALL,DATASETS="aids diabetic heart" report.sbatch
set -eo pipefail
source /data/home/jorge.miguel.ferreira.silva/storage/fedrf_csbj/code/experiments/csbj/slurm/env.sh
cd "$P/code"
python experiments/csbj/analyze.py report --datasets $DATASETS --r-rep 10 --r-rep-reg 4
python experiments/csbj/figures.py
