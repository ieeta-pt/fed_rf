# fed_rf_mk

# Federated Random Forests and XGBoost with PySyft

Privacy-preserving federated learning for tree-based models (RandomForest, XGBoost) using PySyft. Includes orchestration client, datasite server utilities, remote training/evaluation functions, model aggregation, and explainability (SHAP, PFI).
[![PyPI version](https://badge.fury.io/py/fed-rf-mk.svg)](https://badge.fury.io/py/fed-rf-mk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Table of Contents
- What’s New
- Overview
- Repository Structure
- Installation
- Quickstart
- Disclosure ladder
- Reproducing the CSBJ paper
- Detailed Usage
  - Parameters
  - RandomForest example
  - XGBoost example
- Data
- Models & Experiments
- Outputs & Logging
- Extending the Project
- Troubleshooting / FAQ
- Changelog
- Citation / License / Acknowledgements
- To‑Verify

## What’s New
- Modularized architecture:
  - `remote_tasks.py` for remote `ml_experiment` and `evaluate_global_model`.
  - Orchestrator modules for datasites and weights.
  - Handlers for explainability (SHAP, PFI).
  - Aggregation for RF and evaluation-time ensembling for XGB.
- Configurable explainability:
  - Toggle SHAP/PFI and tune `shap_sample_size`, `pfi_n_repeats`.
- Nested hyperparameters:
  - Use `model_params` for model-specific settings (e.g., XGB’s `learning_rate`, `max_depth`, device).
- Safer utilities:
  - Better logging and guards for datasite IO and request status.
- Tests + CI:
  - Unit tests for weights, aggregation, remote tasks, plus pre-commit hooks and GitHub Actions.

## Overview
This project demonstrates a practical federated learning workflow on top of PySyft. It supports:
- Training RF/XGB on multiple datasites without moving raw data.
- Aggregating models (RF) and evaluating ensembles (XGB).
- Explainability across silos via SHAP and PFI with weighted averaging.

High-level flow:

```mermaid
flowchart LR
  A[FLClient] --> B[DataSiteManager]
  B -->|Syft requests| C[(Datasites)]
  C -->|Run| D[remote_tasks.ml_experiment]
  C -->|Run| E[remote_tasks.evaluate_global_model]
  A --> F[WeightManager]
  A --> G[ModelAggregator]
  A --> H[SHAPHandler]
  A --> I[PFIHandler]
  D -->|return model + analysis| G
  D -->|return SHAP| H
  D -->|return PFI| I
  G -->|merged model/ensemble| A
  H -->|averaged SHAP| A
  I -->|averaged PFI| A
```

## Repository Structure
- `package-pysyft/client.py` — FL client orchestrator (connects to datasites, coordinates training/evaluation).
- `package-pysyft/server.py` — server utilities (spawn datasites, manage approvals).
- `package-pysyft/datasites.py` — launch datasites, upload CSV asset, server-side policy toggles.
- `package-pysyft/remote_tasks.py` — remote functions:
  - `ml_experiment(data, dataParams, modelParams)`
  - `evaluate_global_model(data, dataParams, modelParams)`
- `package-pysyft/orchestrator/` — orchestrator modules:
  - `clients.py` — `DataSiteManager` (connect/send requests/check status).
  - `weights.py` — robust `WeightManager` (normalization with None/negative handling).
- `package-pysyft/aggregation/models.py` — `ModelAggregator` (RF estimator merge, XGB ensemble members).
- `package-pysyft/handlers/` — `SHAPHandler`, `PFIHandler`.
- `package-pysyft/analysis/` — notebooks and pipelines, e.g., `2-testing_aids_clinical.ipynb`.
- `tests/` — unit tests for weights, aggregation, remote tasks.
- `.github/workflows/ci.yml` — lint + tests CI.
- `pyproject.toml` — build config and dependencies.

## Installation
Requirement: Python 3.10+

Install dev (recommended for running examples and tests):
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install .[dev]
pre-commit install
```

Minimal install:
```bash
pip install .
```

Key runtime deps:
- syft==0.9.1
- pandas>=2.0
- scikit-learn>=1.3
- cloudpickle>=3.0
- shap>=0.44
- xgboost>=2.0

## Quickstart
Start two train datasites and one eval datasite (example ports/paths). Each
data owner gives the new admin password and the password of the data
scientist account (arguments or the environment variables
`FEDRF_ADMIN_PASSWORD`, `FEDRF_DS_PASSWORD`; `FEDRF_DS_EMAIL` defaults to
`fedlearning@rf.com`). Requests are approved by hand unless `auto_accept=True`,
and the mock that the data scientist sees is synthetic unless a `mock_path` is
given.

```python
from fed_rf_mk.server import FLServer
import threading

servers = [
  FLServer("silo1", 8080, "train_datasets/aids_clinical/part_0.csv", auto_accept=True, analysis_allowed=True),
  FLServer("silo2", 8081, "train_datasets/aids_clinical/part_1.csv", auto_accept=True, analysis_allowed=True),
  FLServer("eval_silo", 8082, "train_datasets/aids_clinical/part_2.csv", auto_accept=True, analysis_allowed=True),
]
for s in servers:
    threading.Thread(target=s.start, daemon=True).start()
```

Train and evaluate with the client:

```python
import os
from fed_rf_mk.client import FLClient

password = os.environ["FEDRF_DS_PASSWORD"]
rf_client = FLClient()
rf_client.add_train_client("silo1", "http://localhost:8080", "fedlearning@rf.com", password)
rf_client.add_train_client("silo2", "http://localhost:8081", "fedlearning@rf.com", password)
rf_client.add_eval_client("eval_silo", "http://localhost:8082", "fedlearning@rf.com", password)

dataParams = {
  "target": "cid",
  "ignored_columns": ["cid"]
}

# RandomForest example (top-level hyperparams for RF)
modelParams = {
  "model": None,
  "model_type": "rf",
  "n_base_estimators": 100,
  "n_incremental_estimators": 20,
  "train_size": 0.8,
  "test_size": 0.8,
  "fl_epochs": 1,
  "site_weights": "size",      # equal (default), size, positives, or manual (weights of add_train_client)
  "release_level": "all",      # 0, 1, 2, 3, 3+1, 3+2 or all (see fed_rf_mk/release.py)
}

rf_client.set_data_params(dataParams)
rf_client.set_model_params(modelParams)

rf_client.send_request()            # values of dataParams and modelParams are bound in each request
rf_client.run_model()               # waits until each training request is approved
results = rf_client.run_evaluate()  # sends the evaluation request, waits for its approval
print(results)
```

## Disclosure ladder

The disclosure ladder (`fed_rf_mk/ladder.py`) orders what the training sites
release, from least to most:

* **Level 0.** One training site, the source, releases its fitted model, used
  as it is at a separate deployment site.
* **Level 1.** Every site also releases its counts n_k and n_k^+, and the source
  model is shifted to the pooled prevalence π = Σ n_k^+ / Σ n_k
  (`shift_prob` for Random Forest probabilities, `shift_logit` for log odds).
* **Level 2.** Instead, the deployment site recalibrates the source model on
  its own labeled patients (`fit_recalibration`, `apply_recalibration`). The
  labeled records never leave it.
* **Level 3.** Every site releases its model, and the K models are combined.
  The federated model is used as it is (3), with the correction of level 1
  (3+1), or with the correction of level 2 (3+2).
  * Random Forest: `site_weights`, `allocate_trees` (largest remainder) and
    `select_trees` with a seeded numpy generator, uniform or by out of bag
    score.
  * XGBoost: the weighted average of the site log odds (`xgb_federated`).

`ladder.py` imports only numpy, scipy, scikit-learn and xgboost, so it can be
used without PySyft. The simulation of the CSBJ paper calls it directly
(`experiments/csbj/run_task.py`). The PySyft path calls the same functions at
the sites, at the coordinator and at the deployment site.

### What leaves each site: `release_level`

`model_params["release_level"]` decides what leaves the training sites
(`fed_rf_mk/release.py`). Levels 0, 1 and 2 also need
`model_params["source_site"]`. The plan of each site is bound in its code
request, so the data owner approves exactly what will be released. A site with
no role at a level gets no request.

| release_level | source site | every other site | the deployment site computes |
|---|---|---|---|
| `0` | model | nothing (no request) | source as it is |
| `1` | model, n_k, n_k^+ | n_k, n_k^+ (`site_counts`, no model trained) | source as it is and shifted to π |
| `2` | model | nothing (no request) | source as it is and recalibrated |
| `3` | model | model | every source as it is, federated model |
| `3+1` | model, n_k, n_k^+ | model, n_k, n_k^+ | as 3, plus level 1 for every source and the federated model |
| `3+2` | model | model | as 3, plus level 2 for every source and the federated model |
| `all` (default) | model, n_k, n_k^+ and the full result of the single round | the same | every level |

* **Counts for the site weights.** At `3` and `3+2`, a site also sends n_k with
  `size` weights and n_k^+ with `positives` weights. With `equal` or `manual`
  weights it sends no count.
* **Out of bag scores.** They leave a site only for Random Forest with
  `tree_sampling="oob"`, at `3`, `3+1` and `3+2`.
* **Analysis outputs.** SHAP and PFI outputs leave only with `all`.
* **What the deployment site receives.** It gets only what the level released:
  * the source model for `0`, `1` and `2`;
  * every model with the site weights and the tree selection for `3`, `3+1` and
    `3+2`;
  * π and π_k for `1` and `3+1`.

  It computes only the levels it is given.
* **`all`.** It keeps the single round of earlier versions and is meant for
  experiments.

`experiments/csbj/tests/test_pysyft_release.py` checks, for every value,
which keys leave each site and what the deployment site receives. The observed
keys are listed in `experiments/csbj/results/pysyft_report.md`.

### Options in `model_params`

| option | default | meaning |
|---|---|---|
| `release_level` | `all` | `0`, `1`, `2`, `3`, `3+1`, `3+2` or `all` |
| `source_site` | None | name of the source site, for `0`, `1` and `2` |
| `site_weights` | `equal` | `equal`, `size`, `positives`, or `manual` (weights given to `add_train_client`) |
| `tree_sampling` | `uniform` | `uniform` or `oob` (Random Forest) |
| `seed` | 42 | int or list of ints, generator of the tree selection |
| `site_seeds` | None | {site name: random_state of its local model}; otherwise 42 at every site |
| `calibration` | `fraction` 0.2, `n_cal` None, `random_state` 42 | calibration part of the deployment site for level 2 |
| `return_predictions` | False | also return the per record outputs of the test part of the deployment site |
| `approval_timeout` | 300 | seconds that `run_model` and `run_evaluate` wait for an approval |

### Approval, mock data and accounts

* **Approval.** Every DataSite runs submitted code only after its data owner
  approves it. `FLServer` approves by hand by default (`auto_accept=False`),
  through `list_pending_requests`, `inspect_request`, `request_parameters`
  and `approve_request`.
* **Bound values.** The values of `dataParams` and `modelParams` are bound in
  each request (`sy.Constant`). An approval therefore covers these values, and
  other values need a new request.
* **Order of the requests.**
  * `send_request` sends the training requests.
  * The evaluation request carries the released models, so it is sent after
    training, by `send_evaluation_request` or by `run_evaluate`.
* **Mock data.** The data scientist never sees the real data as the mock. Without
  a `mock_path`, each DataSite builds a synthetic mock with 100 rows and the
  columns and dtypes of its data. Values are drawn independently for each
  column with a fixed seed.
* **Accounts.**
  * The data owner gives the new admin password and the password of the data
    scientist account, as arguments or in `FEDRF_ADMIN_PASSWORD` and
    `FEDRF_DS_PASSWORD`. `FEDRF_DS_EMAIL` defaults to `fedlearning@rf.com`.
  * The PySyft default admin password is replaced at start.
  * Guest signup is disabled unless asked for (`allow_guest_signup`, or
    `FEDRF_ALLOW_GUEST_SIGNUP=1`).
* **Further detail.** What each step releases, where it runs and what the code
  does not cover is described in `experiments/csbj/results/pysyft_report.md`.

## Reproducing the CSBJ paper

Every table, figure and number of the CSBJ paper comes from
`experiments/csbj`. The full instructions are in `experiments/csbj/README.md`.

1. **Environment.** Python 3.12 or newer (the lock file pins numpy 2.5.5, scipy
   1.18.1 and xgboost 3.4.1, which need 3.12), then
   `pip install -r experiments/csbj/requirements_lock.txt`. The simulation does
   not need PySyft.
2. **Data.**
   * `python experiments/csbj/fetch_data.py` fetches AIDS (UCI 890) and
     Diabetic Retinopathy Debrecen (UCI 329).
   * Heart (`heart_2022_no_nans.csv`, Kaggle
     `kamilpytlak/personal-key-indicators-of-heart-disease`) is downloaded by
     hand into the data folder.
   * Then `python experiments/csbj/data.py` encodes Heart.
3. **Tests.** `python -m pytest -q fed_rf_mk/tests/test_ladder.py experiments/csbj/tests`.
4. **Runs.**
   * `experiments/csbj/run_task.py` runs one task (dataset, repetition, fold).
   * The paper uses 10 repetitions of 5 fold cross validation (50 draws) for
     the main grid, and 4 repetitions for the regularized sensitivity
     configuration.
   * On a Slurm cluster, `experiments/csbj/slurm/submit.sh` submits them as
     chained arrays.
5. **Tables, numbers and figures.**
   * `python experiments/csbj/analyze.py report --datasets aids diabetic heart --r-rep 10 --r-rep-reg 4`
     writes the tables, `numbers.json` and `numbers.md`.
   * `python experiments/csbj/figures.py` writes the figures.
   * `python experiments/csbj/si_tables.py` writes the supplementary tables.
6. **Results in this repository.** The results of the paper are committed in
   `experiments/csbj/results`, so every number can be traced to a commit.
7. **PySyft path.** `experiments/csbj/tests/test_pysyft_equivalence.py` checks
   that the PySyft path gives the predictions of the simulation. It needs a
   separate environment (`experiments/csbj/requirements_pysyft_lock.txt`,
   because PySyft 0.9.5 needs numpy < 1.27).

## Detailed Usage

### Parameters

Data Parameters (dataParams)

Name | Type | Default | Description | Example
---|---|---|---|---
target | str | required | Target column name | "cid"
ignored_columns | list[str] | [] | Columns excluded from features | ["cid","id"]

Top-level Model Parameters (modelParams)

Name | Type | Default | Description | Example
---|---|---|---|---
model | bytes or None | None | Serialized seed model (None for cold start) | None
model_type | str | "rf" | "rf" or "xgb" | "xgb"
n_base_estimators | int | required | Base number of trees | 200
n_incremental_estimators | int | 0 | Extra trees for warm-start | 50
train_size | float | 0.8 | Train split ratio on each site | 0.7
test_size | float | 0.2 | Eval split ratio (use 1.0 for eval silos) | 1.0
fl_epochs | int | 1 | Global rounds | 1
allow_analysis | bool | False | Gate SHAP/PFI analysis (and server policy) | True
analysis | dict | {} | Fine-grained analysis config (see below) | {"enabled":True,...}
model_params | dict | {} | Model-specific hyperparameters (nested) | {"max_depth":6,...}


RandomForest Hyperparameters (modelParams)

Name | Type | Default | Description | Example
---|---|---|---|---
criterion | str | "gini" | Split criterion | "entropy"
max_depth | int or None | None | Max tree depth | 12
min_samples_split | int | 2 | Min samples to split | 2
min_samples_leaf | int | 1 | Min samples per leaf | 1
max_features | str or int | "sqrt" | Features per split | "sqrt"
bootstrap | bool | True | Bootstrap samples | True
n_jobs | int or None | None | Threads | -1
… | … | … | Additional sklearn RF kwargs supported | …

XGBoost Hyperparameters (modelParams)

Name | Type | Default | Description | Example
---|---|---|---|---
device | str | "cpu" | "cpu" or "cuda" | "cuda"
verbosity | int | 1 | 0–3 | 1
validate_parameters | bool | True | Validate input params | True
disable_default_eval_metric | bool | False | Disable default metric | False
learning_rate | float | 0.1 | Eta | 0.05
max_depth | int | 6 | Max depth | 8
min_child_weight | float | 1 | Min child weight | 1
gamma | float | 0 | Min split loss | 0
subsample | float | 1.0 | Row subsampling | 0.8
colsample_bytree | float | 1.0 | Col subsampling per tree | 0.8
colsample_bylevel | float | 1.0 | Col subsampling per level | 1.0
colsample_bynode | float | 1.0 | Col subsampling per node | 1.0
reg_lambda | float | 1.0 | L2 regularization | 1.0
reg_alpha | float | 0.0 | L1 regularization | 0.0
tree_method | str | "auto" | "hist","approx","exact","auto" | "hist"
max_delta_step | float | 0 | Max delta step | 0
scale_pos_weight | float | 1.0 | Class imbalance control | 1.0
booster | str | "gbtree" | Booster type | "gbtree"
grow_policy | str | "depthwise" | "depthwise" or "lossguide" | "depthwise"
max_leaves | int | 0 | Max leaves | 0
max_bin | int | 256 | Histogram bins | 256
sampling_method | str | "uniform" | "uniform" or "gradient_based" | "uniform"
Optional | various | N/A | `updater`, `refresh_leaf`, `process_type`, `num_parallel_tree`, `monotone_constraints`, `interaction_constraints`, `multi_strategy` | see XGB docs

### RandomForest example

```python
from package_pysyft.client import FLClient
rf_client = FLClient()

# connect clients (see Quickstart)
# ...

dataParams = {"target": "cid", "ignored_columns": ["cid"]}
modelParams = {
  "model": None,
  "model_type": "rf",
  "n_base_estimators": 150,
  "n_incremental_estimators": 30,
  "train_size": 0.8,
  "test_size": 0.8,
  "fl_epochs": 1,
  "allow_analysis": True,
  "max_depth": 10, 
  "max_features": "sqrt", 
  "bootstrap": True, 
}
rf_client.set_data_params(dataParams)
rf_client.set_model_params(modelParams)
rf_client.run_model()
print(rf_client.run_evaluate())
```

### XGBoost example

```python
from package_pysyft.client import FLClient
xgb_client = FLClient()

# connect clients (see Quickstart)
# ...

dataParams = {"target": "cid", "ignored_columns": ["cid"]}
modelParams = {
    "model": None,
    "model_type": "xgb",
    "n_base_estimators": 200,
    "n_incremental_estimators": 50,
    "train_size": 0.8,
    "test_size": 0.8,
    "fl_epochs": 1,
    "allow_analysis": True,
    "device": "cpu",  # or "cuda"
    "learning_rate": 0.1,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "verbosity": 1,
    "validate_parameters": True

}
xgb_client.set_data_params(dataParams)
xgb_client.set_model_params(modelParams)
xgb_client.run_model()
print(xgb_client.run_evaluate())
```

## Data
- Input: CSV files per datasite with consistent schema.
- `dataParams["target"]` must exist in all files; `ignored_columns` are dropped from features.
- `datasites.py` uploads a single asset called "Asset" by default (first dataset/asset is used in examples).

## Models & Experiments
- RandomForest:
  - Training: local RF per silo; aggregation takes t_k trees per site by largest remainder (`ladder.allocate_trees`), chosen with a seeded generator (`ladder.select_trees`).
  - Warm-start: `n_incremental_estimators` grows trees for subsequent rounds.
- XGBoost:
  - Training: XGB per silo; for continuation, the booster is reused and `n_estimators` is increased.
  - Evaluation: weighted margin (logit) averaging of ensemble members; picks best-weight seed for next round.

Explainability:
- SHAP: mean absolute SHAP per feature; averaged across silos using normalized weights.
- PFI: permutation importance on eval split; mean/std per feature; averaged across silos.

## Outputs & Logging
- Remote training returns:
  - `"model"` (serialized), `"model_type"`, sizes, and optionally `"shap_data"`, `"pfi_data"`.
- Client logs progress; datasites log request approvals. Arrays in SHAP/PFI are normalized to Python floats for downstream use.

## Extending the Project
- Add new model types by:
  - Updating `remote_tasks.ml_experiment` to construct and train the model.
  - Extending `ModelAggregator` for aggregation/ensembling semantics.
- Add new explainability methods through handlers similar to SHAP/PFI.

## Troubleshooting / FAQ
- Evaluate fails with “Reference don’t match: dict”
  - The evaluation request is sent after training (`send_evaluation_request` or `run_evaluate`) and must be approved by the data owner of the evaluation datasite before it runs.
- Missing packages
  - Install with `pip install .[dev]` to get all deps (including shap/xgboost).
- SHAP/PFI slow
  - Reduce `analysis.shap_sample_size` or `analysis.pfi_n_repeats`, or disable via `analysis.enabled=False`.

## Changelog
See `CHANGELOG.md` for 1.1.0 and later.

- 0.2.0 (2025-09-06)
  - Modular orchestrator/handlers/aggregation
  - Nested `model_params` support for RF/XGB
  - Explainability controls (SHAP/PFI) and parameters
  - Safer datasite loading + logging, fallback eval registration
  - Tests + CI, pre-commit hooks

## Citation / License / Acknowledgements
- Citation: see `CITATION.cff`.
- License: MIT
- Built with:
  - [PySyft](https://github.com/OpenMined/PySyft)
  - [scikit-learn](https://scikit-learn.org/)
  - [XGBoost](https://xgboost.readthedocs.io/)
  - [SHAP](https://shap.readthedocs.io/)
