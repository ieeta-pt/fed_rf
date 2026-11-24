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

Paths are set by the environment variables `CSBJ_DATA`, `RUNS`, `PREDS` and
`CSBJ_RESULTS` (see `config.py`).

Heart encoding, distinct values, cache:

    python experiments/csbj/data.py

This writes `heart_encoded.npz` to the data folder and `heart_encoding.csv`
(column, kind, value, code) to the results folder.

## Tests

    python -m pytest -q fed_rf_mk/tests/test_ladder.py experiments/csbj/tests

`fed_rf_mk/tests/test_remote_tasks.py` needs PySyft and is not part of this setup.

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
* Level 2 rows rank records by a z + b for AUROC and AUPRC. With a > 0 this
  is the ranking of z, also where sigma(a z + b) rounds to exactly 0 or 1
  (this happens with small calibration sets that separate the classes).
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
