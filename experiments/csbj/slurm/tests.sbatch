#!/bin/bash
# Unit tests on a compute node.
#SBATCH --job-name=fedrf_tests
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/data/home/jorge.miguel.ferreira.silva/storage/fedrf_csbj/logs/%x_%j.out
set -eo pipefail
source /data/home/jorge.miguel.ferreira.silva/storage/fedrf_csbj/code/experiments/csbj/slurm/env.sh
cd "$P/code"
python experiments/csbj/data.py | grep -v "^  [A-Z].*: \["
python -m pytest -q -p no:cacheprovider fed_rf_mk/tests/test_ladder.py experiments/csbj/tests
