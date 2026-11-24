"""Configuration of the CSBJ disclosure ladder experiments (E1 and variants).

Paths come from environment variables, with defaults relative to the folder
that holds the repository (``<repo>/..``):

    CSBJ_DATA     input data             default <repo>/../data
    RUNS          metric run files       default <repo>/../runs
    PREDS         per row predictions    default <repo>/../preds
    CSBJ_RESULTS  tables, figures, json  default <repo>/../results
"""
import os
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO.parent


def _path(var, default):
    return Path(os.environ.get(var, str(BASE / default))).resolve()


def data_dir():
    return _path("CSBJ_DATA", "data")


def runs_dir():
    return _path("RUNS", "runs")


def preds_dir():
    return _path("PREDS", "preds")


def results_dir():
    return _path("CSBJ_RESULTS", "results")


DATASETS = ("aids", "diabetic", "heart")
DS_CODE = {"aids": 1, "diabetic": 2, "heart": 3}

K_LIST = tuple(range(2, 11))

# partition name -> (part_code, Dirichlet alpha or None)
PARTITIONS = {"equal": (0, None), "dir0.5": (5, 0.5), "dir0.1": (1, 0.1)}
MIN_PER_CLASS = 5  # records of each class given to every site before the Dirichlet split

R_REP = 4  # repetitions of 5 fold CV, R = 5 * R_REP draws
N_FOLDS = 5
CAL_FRACTION = 0.2  # share of the deployment site used for level 2 calibration

LEARNER_CODE = {"rf": 1, "xgb": 2, "lr": 3}
CENTRAL_PART_CODE = 99

WEIGHT_CODE = {"equal": 0, "size": 1, "positives": 2}
SAMPLING_CODE = {"uniform": 0, "oob": 1}
DEFAULT_WEIGHTS = "equal"
DEFAULT_SAMPLING = "uniform"

TREE_SELECT_TAG = 7
SHIFT_TAG = 8
NCAL_TAG = 9
SHIFT_CODE = {"shift_half": 1, "shift_double": 2}
N_CAL_LIST = (25, 50, 100, 200)

RF_PARAMS = dict(n_estimators=100, n_jobs=1)
XGB_PARAMS = dict(n_estimators=100, max_depth=6, learning_rate=0.3,
                  tree_method="hist", n_jobs=1)
LR_PARAMS = dict(max_iter=5000)

# Heart encoding: ordered categories in their natural order, codes 0, 1, ...
HEART_TARGET = "HadHeartAttack"
HEART_BINARY_EXTRA = {"Sex": {"Female": 0, "Male": 1}}
HEART_ORDERED = {
    "GeneralHealth": ["Poor", "Fair", "Good", "Very good", "Excellent"],
    "LastCheckupTime": [
        "Within past year (anytime less than 12 months ago)",
        "Within past 2 years (1 year but less than 2 years ago)",
        "Within past 5 years (2 years but less than 5 years ago)",
        "5 or more years ago",
    ],
    "RemovedTeeth": ["None of them", "1 to 5", "6 or more, but not all", "All"],
    "SmokerStatus": [
        "Never smoked",
        "Former smoker",
        "Current smoker - now smokes some days",
        "Current smoker - now smokes every day",
    ],
    "ECigaretteUsage": [
        "Never used e-cigarettes in my entire life",
        "Not at all (right now)",
        "Use them some days",
        "Use them every day",
    ],
    "AgeCategory": [
        "Age 18 to 24", "Age 25 to 29", "Age 30 to 34", "Age 35 to 39",
        "Age 40 to 44", "Age 45 to 49", "Age 50 to 54", "Age 55 to 59",
        "Age 60 to 64", "Age 65 to 69", "Age 70 to 74", "Age 75 to 79",
        "Age 80 or older",
    ],
}
HEART_NOMINAL = ("State", "RaceEthnicityCategory", "HadDiabetes",
                 "TetanusLast10Tdap", "CovidPos")


def seed_for(*ints):
    """Deterministic 32 bit seed derived from a tuple of integers."""
    return int(np.random.SeedSequence([int(i) for i in ints]).generate_state(1)[0])
