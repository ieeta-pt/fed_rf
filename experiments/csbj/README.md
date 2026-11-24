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
