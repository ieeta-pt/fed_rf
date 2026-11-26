# Environment shared by every job of the project. Sourced, not executed.
P=/data/home/jorge.miguel.ferreira.silva/storage/fedrf_csbj
export PIP_CACHE_DIR=$P/.cache/pip
export XDG_CACHE_HOME=$P/.cache
export MPLCONFIGDIR=$P/.cache/matplotlib
export TMPDIR=${TMPDIR:-$P/.cache/tmp}
export JOBLIB_TEMP_FOLDER=$P/.cache/joblib
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export CSBJ_DATA=$P/data
export RUNS=$P/runs
export RUNS_REGULARIZED=$P/runs_regularized
export PREDS=$P/preds
export CSBJ_RESULTS=$P/results
mkdir -p "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$TMPDIR" "$JOBLIB_TEMP_FOLDER"
module load python/3.12.12
source $P/venv/bin/activate
