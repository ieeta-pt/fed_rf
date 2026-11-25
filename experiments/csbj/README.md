# CSBJ disclosure ladder experiments (E1 and variants)

Base commit of ieeta-pt/fed_rf: `117e5d5` (Merge pull request #5 from ieeta-pt/0.0.7).

The simulation uses only `fed_rf_mk/ladder.py` (numpy, scipy, scikit-learn,
xgboost). It does not import PySyft.

## Environment

Python 3.10 or newer, then

    pip install -r experiments/csbj/requirements_lock.txt

## Data

    python experiments/csbj/fetch_data.py

* AIDS Clinical Trials Group Study 175, UCI dataset 890, 2,139 rows, target `cid` (521 positives).
* Diabetic Retinopathy Debrecen, UCI dataset 329, 1,151 rows, target `Class` (611 positives).
  Read from the original ARFF file, because the CSV served through `ucimlrepo` names
  two columns `exudate3` and repeats the values of `exudate3` in the second one.
  The file has 5 exact duplicate rows, all kept.
* Heart, Kaggle `kamilpytlak/personal-key-indicators-of-heart-disease`, file
  `heart_2022_no_nans.csv` (246,022 rows, 40 columns), downloaded by hand into the data folder.
  `HadAngina` is left out of the features, because angina or coronary heart disease is close
  to a restatement of the target (a heart attack is usually a manifestation of coronary heart
  disease). Heart keeps 38 features.

Paths are set by the environment variables `CSBJ_DATA`, `RUNS`, `PREDS` and
`CSBJ_RESULTS` (see `config.py`).

Heart encoding, distinct values, cache:

    python experiments/csbj/data.py

This writes `heart_encoded.npz` to the data folder and `heart_encoding.csv`
(column, kind, value, code) to the results folder.

## Tests

    python -m pytest -q fed_rf_mk/tests/test_ladder.py experiments/csbj/tests

`fed_rf_mk/tests/test_remote_tasks.py` needs PySyft and is not part of this setup.

### PySyft path

PySyft 0.9.5 needs numpy < 1.27, so it gets its own environment (versions in
`requirements_pysyft_lock.txt`), next to the E1 one:

    uv venv --python 3.12 ../.venv_syft
    uv pip install --python ../.venv_syft/bin/python -r experiments/csbj/requirements_pysyft_lock.txt
    (cd fed_rf_mk && ../../.venv_syft/bin/python -m pytest -q tests)
    PYSYFT_EQUIV_OUT=$PWD/experiments/csbj/results/pysyft_equivalence.json \
        ../.venv_syft/bin/python -m pytest -q -s experiments/csbj/tests/test_pysyft_equivalence.py

    PYSYFT_RELEASE_OUT=$PWD/experiments/csbj/results/pysyft_release.json \
        ../.venv_syft/bin/python -m pytest -q -s experiments/csbj/tests/test_pysyft_release.py

The equivalence test starts local DataSites (`FLServer`, `auto_accept=True`)
and checks that the PySyft path gives the predictions of the simulation (about
one minute). The release test approves every request by hand and checks what
leaves each site for every `release_level` (about three minutes). Both are
skipped in the E1 environment. Results and the description of
what each site sends are in `results/pysyft_report.md`.

## One task

    RUNS=... PREDS=... python experiments/csbj/run_task.py --dataset aids --task-id 0

A task is one (dataset, r, f), with r = id // 5 and f = id % 5. It trains the
centralized models and runs the 27 cells (3 partitions times K = 2..10) with
`SLURM_CPUS_PER_TASK` joblib workers. Finished cells are skipped unless
`--force` is given. A failing cell writes `<part>_K<K>.error.txt` and the task
exits with code 1.

Smoke run used on the Mac (not used for the results):

    export RUNS=../runs_local/runs PREDS=../runs_local/preds
    python experiments/csbj/run_task.py --dataset aids --rep 0 --fold 0 --K 2 5 10
    python experiments/csbj/run_task.py --dataset diabetic --rep 0 --fold 0 --K 2 5 10
    python experiments/csbj/run_task.py --dataset heart --rep 0 --fold 0 --K 2 10 --subsample 20000

## Implementation notes

* Metric rows. `T` is the number of Random Forest trees averaged, for XGBoost
  the number of trees in the combined model (100 per site with a nonzero
  weight), 0 for logistic regression. For federated rows `n_k` and `pos_k` are
  the totals of the training pool and `pi_k` is the prevalence the combined
  model was trained at (sum_k (t_k/T) pi_k for rf, sum_k w_k pi_k for xgb).
  `n_cal` is 0 for rows that use no calibration records.
* Level 2 rows rank records for AUROC and AUPRC by sign(a) times the input of
  the recalibration (the clipped probability for rf, the log odds otherwise).
  This is the ranking of sigma(a z + b) in exact arithmetic, also where
  sigma(a z + b) rounds to exactly 0 or 1 (small calibration sets that
  separate the classes), and it does not depend on the last bit of logit.
* Reproducibility. A cell run twice on the same machine gives identical
  files, also across the two Pleiades CPU nodes. Between an arm64 Mac and the
  x86-64 nodes, XGBoost and logistic regression agree, while a few Random
  Forest splits differ (near ties in the impurity decrease resolved
  differently in floating point), so the Mac smoke numbers are not the
  reference. The reported numbers come from Pleiades.
* Every weight and sampling configuration (E5) is scored on `test`. The
  default configuration (equal weights, uniform sampling) is also scored on
  `shift_half` and `shift_double` (E10) and on the calibration subsets (E12).
  `fed_L1g` is scored on `test` only.
* E10 removes floor(n/2) of the positives (`shift_half`) or of the negatives
  (`shift_double`) of the calibration part and of the test part.
* E12 keeps a size only when the calibration part has enough records of each class.
* `central_sites.parquet` holds the leaf statistics of the centralized models (E9).
* Peak memory per cell is VmHWM reset at the start of the cell on Linux, and
  the high water mark of the worker process on macOS.
* XGBoost with the default `min_child_weight=1` cannot have a leaf with a
  single training row, because the hessian p(1-p) of a row is at most 0.25,
  so its single record leaf fraction is 0 by construction.

## Pleiades

Everything lives under `P=/data/home/jorge.miguel.ferreira.silva/storage/fedrf_csbj`
(code, data, venv, runs, runs_regularized, preds, logs, results, .cache).
`slurm/env.sh` sets the caches under `$P/.cache`, single threaded BLAS and the
run folders, loads `python/3.12.12` and activates `$P/venv`.

    module load python/3.12.12 && python -m venv $P/venv && source $P/venv/bin/activate
    pip install -r experiments/csbj/requirements_lock.txt
    sbatch --nice=100 experiments/csbj/slurm/tests.sbatch             # tests and Heart encoding
    bash experiments/csbj/slurm/submit.sh --timing                     # task 0 of each dataset
    bash experiments/csbj/slurm/submit.sh --r-rep 10                   # main grid, 50 draws
    bash experiments/csbj/slurm/submit.sh --r-rep 4 --hp regularized --after <last job id>
    python experiments/csbj/slurm/status.py --r-rep 10                 # or --r-rep 4 --hp regularized
    sbatch --nice=100 --export=ALL,DATASETS="aids diabetic heart" experiments/csbj/slurm/report.sbatch

`submit.sh --datasets heart` resubmits one dataset only.

`submit.sh` chains one array per dataset (`%1` inside each array, `afterany`
between arrays, `--nice=100`), so at most one task of the project runs at any
time. Tasks are requeue safe: finished cells are skipped.

## Regularized sensitivity configuration

`--hp regularized` uses RF `min_samples_leaf=5` and XGBoost `learning_rate=0.1`
(everything else unchanged, same seeds), scores only the main arms (equal
weights, uniform sampling, eval `test`, full calibration part, plus `fed_L1g`)
and writes to `RUNS_REGULARIZED`. `analyze.py compare-hp` puts the shares, the
AUROC gain of level 3, the centralized models and the single record leaf
fractions of the two configurations side by side, on the draws the two grids
share (r = 0..3, 20 draws). With `min_samples_leaf=5` scikit-learn counts distinct
bootstrap rows, so every leaf holds at least 5 distinct rows, with one exception:
a tree whose bootstrap sample has fewer than 10 distinct rows cannot be split and
is a single leaf holding all of them. This happens at sites with 10 to 12 records
(the 5 + 5 minimum of the Dirichlet partitions), where a bootstrap sample of 10
draws can hold only 4 distinct rows.

## Tables, numbers and figures

    python experiments/csbj/analyze.py report --datasets aids diabetic heart --r-rep 10 --r-rep-reg 4
    python experiments/csbj/figures.py

`report` checks that every expected output is present with no error file, then
writes into `CSBJ_RESULTS` the tables T1 to T3 and the supplementary tables (CSV
and booktabs LaTeX), `numbers.json` and `numbers.md` (every value with its
interval and one line on how it was computed). `figures.py` reads only those CSV
files and writes PDF and PNG figures into `CSBJ_RESULTS/figures`. The logistic
regression learner (E13) is run but not reported. On Pleiades, `slurm/report.sbatch`
runs both (about 40 minutes for the three datasets, mostly reading the run files), and only `$P/results` is
copied back:

    rsync -av pleiades:/data/home/jorge.miguel.ferreira.silva/storage/fedrf_csbj/results/ "$CSBJ/results/"

