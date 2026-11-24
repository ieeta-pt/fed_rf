#!/bin/bash
# Submit the E1 arrays, one per dataset, chained so that at most one task of
# the project runs at any time (%1 inside an array, afterany between arrays),
# all with --nice=100.
#
#   bash submit.sh [--r-rep 4] [--hp default|regularized] [--timing] [--after JOBID]
#                  [--datasets "aids diabetic heart"]
#
#   --timing   only task 0 of each dataset
#   --after    the first array waits for this job (e.g. the last array of the main grid)
#
# Prints the job ids, the last one on the final line.
set -euo pipefail
R_REP=4
HP=default
TIMING=0
AFTER=""
DATASETS="aids diabetic heart"
while [ $# -gt 0 ]; do
  case "$1" in
    --r-rep) R_REP=$2; shift 2 ;;
    --hp) HP=$2; shift 2 ;;
    --timing) TIMING=1; shift ;;
    --after) AFTER=$2; shift 2 ;;
    --datasets) DATASETS=$2; shift 2 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done
HERE=$(cd "$(dirname "$0")" && pwd)
LAST_ID=$((5 * R_REP - 1))
prev="$AFTER"
for DS in $DATASETS; do
  case "$DS" in
    heart) RES="--cpus-per-task=9 --mem=16G --time=00:30:00" ;;
    *)     RES="--cpus-per-task=9 --mem=4G --time=00:15:00" ;;
  esac
  if [ "$TIMING" = 1 ]; then ARRAY="0"; else ARRAY="0-${LAST_ID}%1"; fi
  DEP=""
  if [ -n "$prev" ]; then DEP="--dependency=afterany:${prev}"; fi
  jid=$(sbatch --parsable --nice=100 $RES $DEP --array="$ARRAY" \
        --job-name="fedrf_${DS}_${HP}" --export=ALL,DS="$DS",HP="$HP" "$HERE/task.sbatch")
  echo "$DS $HP array=$ARRAY job=$jid ${DEP}"
  prev=$jid
done
echo "$prev"
