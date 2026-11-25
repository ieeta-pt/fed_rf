# PySyft path of fed_rf and the disclosure ladder

Facts only, each with the file and function where it happens. Paths are
relative to the repository root. The checks were run on the Mac on 2026-11-25.

## 1. Versions

* PySyft 0.9.5 (`setup.py` asks for `syft>=0.9.5,<0.9.6`).
* PySyft 0.9.5 requires numpy < 1.27, pandas 2.2.2, pyarrow 17.0.0 and
  matplotlib < 3.9.1. The E1 environment (Python 3.14.10, numpy 2.5.5,
  scipy 1.18.1) cannot hold it. The PySyft checks therefore use a separate
  environment, `../.venv_syft`: Python 3.12.14, numpy 1.26.4, scipy 1.17.1,
  scikit-learn 1.9.1, xgboost 3.4.1 and pandas 2.2.2. The exact versions are
  in `experiments/csbj/requirements_pysyft_lock.txt`. The E1 environment was
  not changed.
* `import syft` failed until `sniffio` was installed by hand, because
  `kr8s` needs it and syft 0.9.5 does not list it.
* The simulation side of the equivalence test runs in the same environment as
  PySyft. In addition, `run_task.run_cell` was run in both environments for the
  two test configurations, and the outputs were compared.
  * Per record predictions of RF and XGBoost differ by at most 3.3e-16.
  * Predictions of logistic regression (not on the PySyft path) differ by at
    most 2.0e-13.
  * Metric rows (AUROC, Brier, MCC at π, a, b) differ by at most 2.5e-13. The
    largest differences are in the logistic regression rows.

## 2. Where each step runs

| step | runs in | code |
|---|---|---|
| training of the site model | each training DataSite, as approved code | `fed_rf_mk/remote_tasks.py`, `ml_experiment` |
| site weights, pooled prevalence, tree allocation and selection | the requester's Python process (the coordinator), not a DataSite | `fed_rf_mk/client.py`, `FLClient.run_model`; `fed_rf_mk/aggregation/models.py`, `ModelAggregator.combine`, `ModelAggregator.merge_estimators` |
| levels 0, 1, 2 and 3 at the deployment site | the evaluation DataSite, as approved code | `fed_rf_mk/remote_tasks.py`, `evaluate_global_model` (inner `_ladder_evaluate`) |

The approved code ran in the server process that `sy.orchestra.launch` starts
(`fed_rf_mk/datasites.py`, `spawn_server`). That process has a different pid
from the requester. The code imports `fed_rf_mk.ladder` inside that process. A
code request carries only the source of the function, so `fed_rf_mk` must be
installed at every DataSite. In the local test, the server process gets the
`sys.path` of the process that launched it.

## 3. What each training site sends, per level

`ml_experiment` runs once per site and returns one dict. The training round is
the same for every level. Every site sends the following.

* `model`: the fitted model, pickled with `cloudpickle` (all T_k trees).
* `model_type`, `n_base_estimators`, `n_incremental_estimators`, `train_size`,
  `test_size`: the parameters it received.
* `sample_size` and `n_train`: n_k, the number of training records.
* `n_pos`: n_k^+, the training records of the positive class
  (`classes_[1]`). It is None when the model does not have two classes.
* `oob_scores`: the out of bag accuracy of each tree
  (`ladder.tree_oob_scores`). Sent only for Random Forest with
  `tree_sampling="oob"`, None otherwise.
* `analysis_status`.
* `shap_data` and `pfi_data`, only when both the request (`allow_analysis`)
  and the site policy (`datasites.is_analysis_allowed`) allow analysis. This is
  unchanged.

The ladder needs the following at each level. The code sends all of it in the
one round above.

| level | what the ladder needs from the training sites | what the code sends |
|---|---|---|
| 0 | the model of the source site | every site: model, n_k, n_k^+ (and out of bag scores for oob sampling) |
| 1 | level 0, plus n_k and n_k^+ of every site | the same |
| 2 | the model of the source site (the labels are at the deployment site) | the same |
| 3, 3 with 1, 3 with 2 | the models of every site (out of bag scores for oob sampling), plus the counts for 3 with 1 | the same |

With `train_size >= 1` (or missing), every record of the site trains its model
and no local test part is kept (`ml_experiment`, `preprocess`). With
`train_size < 1`, the old split with `random_state=42` is used. The local seed
of the model is `modelParams["random_state"]` (default 42).
`FLClient.run_model` sets it per site from `site_seeds`.

## 4. Combination at the coordinator

`FLClient.run_model`, `ModelAggregator.combine` and
`ModelAggregator.merge_estimators` do the following.

* **Site order.** The sites are taken in the order in which they were added
  (`add_train_client`), not in the order in which they finished. The seeded
  selection therefore does not depend on timing.
* **Counts.** n_k and n_k^+ come from the site results
  (`ModelAggregator.counts`). π is computed with `ladder.pooled_prevalence`.
* **Site weights.** `ladder.site_weights`, with `site_weights` set to `equal`,
  `size` or `positives`. `manual` uses the weights given to
  `add_train_client`, normalized by `WeightManager.normalize_weights`.
* **Random Forest.**
  * `ladder.allocate_trees(w, T_k)` decides how many trees each site gives.
  * Then `ladder.select_trees` runs site by site in the order above, all from
    one generator, `numpy.random.default_rng(seed)`. The draw is uniform, or
    in proportion to `oob_scores`.
  * The merged forest (`seed_model`) holds the selected trees in site order.
    `ladder.oob_fallback` records whether a site fell back to uniform.
* **XGBoost.** No trees are merged. The site models are kept with their
  weights as `ensemble_members`. `seed_model` is the member with the largest
  weight (ties to the first site), as before.
* **What goes to the deployment site.** `modelParams["ladder"]`, which holds:
  * `sites`
  * `models`: the K site models, every tree
  * `weights`, `weights_kind`
  * `n`, `n_pos`, `pi`
  * `model_type`
  * for Random Forest also `sampling`, `seed`, `T_k`, `t`, `selections` and
    `oob_fallback`

  It also gets `model` and `ensemble_members`.
* **SHAP and PFI.** They are averaged with the same site weights.

## 5. What the deployment site computes and keeps

`evaluate_global_model` takes the ladder branch when `modelParams["ladder"]` is
set. It runs at the evaluation DataSite and does the following.

* **Records used.** It uses every record of that DataSite.
* **Calibration and test parts.**
  * A stratified calibration part comes from `train_test_split`. Its size is
    `calibration.n_cal` records, or `calibration.fraction` of the site
    (default 0.2). Its seed is `calibration.random_state`.
  * The rest is the test part.
* **Source models, for every source site k.**
  * Level 0 is the model as it is.
  * Level 1 uses `ladder.shift_prob` (RF) or `ladder.shift_logit` (XGBoost),
    with π and π_k = n_k^+/n_k.
  * Level 2 uses `ladder.fit_recalibration` on the calibration part and
    `ladder.apply_recalibration`. For RF, z = logit(`clip_prob`(p, T_k)), as in
    the simulation.
  * RF tree probabilities come from `ladder.rf_tree_probs`. It is a new
    function that stores float32, like `run_task.fit_rf`.
* **Federated model.**
  * Level 3 uses `ladder.rf_federated` with the selections, or
    `ladder.xgb_federated` with the weights.
  * Level 3 with 1 uses `ladder.rf_federated_shift` or
    `ladder.xgb_federated_shift`.
  * The sensitivity analysis (`fed_L1g`) uses `ladder.rf_federated_global_shift`
    or `ladder.xgb_federated_global_shift`.
  * Level 3 with 2 fits `fit_recalibration` on the calibration part. For RF,
    z = logit(`clip_prob`(p, T)).
* **What it returns.**
  * For every arm, AUROC, AUPRC, Brier, MCC at π and MCC at 0.5 on the test
    part. Level 2 arms also return a and b.
  * n_cal, n_test, π and the weights.
  * With `return_predictions=True` (default False), it also returns the per
    record probabilities and log odds of the test part and the positions of the
    test records. It never returns labels. The equivalence test uses this
    option.
* **What it keeps.** The records, their labels and the calibration part stay
  at the deployment site.
* **Not covered by the test.** These metrics are not part of the equivalence
  test. AUROC of level 2 is computed on σ(a z + b) here, while `run_task.py`
  ranks level 2 rows by sign(a) z.

The new options in `model_params` and their defaults are in
`fed_rf_mk/client.py`, `LADDER_DEFAULTS` (filled in by
`FLClient.set_model_params`):

| option | default |
|---|---|
| `site_weights` | `equal` |
| `tree_sampling` | `uniform` |
| `seed` | 42, an int or a list of ints |
| `site_seeds` | None, so every site uses 42 |
| `calibration` | `n_cal` None, `fraction` 0.2, `random_state` 42 |
| `return_predictions` | False |

## 6. How code requests are approved

* **Requests.**
  * `DataSiteManager.send_training_requests` and `send_evaluation_requests`
    (`fed_rf_mk/orchestrator/clients.py`) wrap `ml_experiment` and
    `evaluate_global_model` in `sy.syft_function`, with
    `MixedInputPolicy(client, data=<asset>, dataParams=dict, modelParams=dict)`.
  * Each one is put in a `sy.Project` and sent.
  * The output policy is the syft default, which is `OutputPolicyExecuteOnce`
    as observed.
  * A request holds the source of the function and pins the asset. It holds no
    values of `dataParams` or `modelParams`, which are only typed as `dict`.
    The values are passed later, when `FLClient.run_model` and
    `FLClient.run_evaluate` call `datasite.code.ml_experiment(...)` and
    `datasite.code.evaluate_global_model(...)`.
  * In a check with `auto_accept=True`, the requester called one approved
    function twice with different `dict` values, and both calls ran.
* **Who approves.** The data owner, logged in as the DataSite admin, approves.
  * Manual mode (`FLServer(..., auto_accept=False)`): `FLServer.list_pending_requests`,
    `FLServer.inspect_request(i)` (returns the code) and
    `FLServer.approve_request(i)` (`request.approve(approve_nested=True)`),
    in `fed_rf_mk/server.py`.
  * `auto_accept=True` is the default of `FLServer.__init__`. A thread runs
    `check_and_approve_incoming_requests` (`fed_rf_mk/datasites.py`). Once a
    second, it approves every request that is not approved yet, with
    `approve_nested=True`, without looking at the code. The equivalence test
    uses this mode, as asked.
* **Mock data seen by the requester.**
  * `FLServer.__init__` sets `mock_path = data_path` when no mock is given.
    `create_syft_dataset` (`fed_rf_mk/datasites.py`) then uploads the real data
    as the mock of the asset.
  * Logged in as the data scientist, `asset.mock` was equal to the real data
    in the check.
  * `generate_mock` (`fed_rf_mk/datasets.py`) returns the data unchanged
    (synthetic generation is disabled). `FLServer` never reaches it.
* **Accounts.** `spawn_server` does the following.
  * It logs in with the PySyft default admin account, `info@openmined.org` /
    `changethis`, and syft warns that the password is the default.
  * It enables guest signup.
  * It creates a data scientist account, `fedlearning@rf.com`, with the
    password `****`.
* **Analysis policy.** `FLServer(analysis_allowed=True)` sets
  `datasites.ANALYSIS_ALLOWED` in the process that calls `FLServer.start`.
  Inside the approved code (the server process), `is_analysis_allowed()`
  returned False in the check. So SHAP and PFI at the sites stay off in this
  local setup, whatever `analysis_allowed` says.

## 7. Equivalence with the simulation

The test is `experiments/csbj/tests/test_pysyft_equivalence.py`. The full
output is in `experiments/csbj/results/pysyft_equivalence.json`.

* **Data and partition.** Both configurations use r = 0, f = 0 and the
  partition and seeds of `experiments/csbj/run_task.py`.
  * Every site and the deployment site get a CSV with their records, in the
    order of the simulation. The test checks that the CSV round trip is exact.
  * Each one is served by an `FLServer` with `auto_accept=True`.
  * `FLClient` runs `send_request`, `run_model` and `run_evaluate`.
* **Model settings.**
  * Model seeds come from `site_seeds`, with `seed_for(ds, r, f, part, K, k, learner)`.
  * The tree selection seed is `[ds, r, f, part, K, 7, w, s]`.
  * The calibration part uses `fraction` 0.2 and `random_state` 1000 r + f.
  * XGBoost uses `tree_method="hist"` and `n_jobs=1`, as in `config.XGB_PARAMS`.
    The defaults of `ml_experiment` are `"auto"` and -1.
* **Simulation side.** It is rebuilt from `run_task.fit_rf`, `run_task.fit_xgb`
  and the `ladder` calls of `run_task.run_cell`, in the same order. The test
  first checks this against `run_task.run_cell` itself:
  * the per record predictions of every source arm, and of the default
    federated arms;
  * Brier, AUROC, a and b of the federated metric rows of the configuration.

  The largest difference in the predictions was 0. Brier and AUROC were equal
  within 1e-15, and a and b were exactly equal.
* **Combination.** For both configurations, the test checks that the
  coordinator's n_k, n_k^+, π and weights equal those of the simulation. For
  RF it also checks t_k and the selected tree indices.

| configuration | compared on the test part | levels and sources compared | largest difference | tolerance |
|---|---|---|---|---|
| AIDS, Random Forest, dir0.5, K = 3, size weights, oob sampling | probabilities | src0..2 × L0, L1, L2; fed L0, L1, L1g, L2 (13 arms, 343 records) | 0.0 | 1e-12 |
| Diabetic, XGBoost, dir0.5, K = 4, equal weights | log odds | src0..3 × L0, L1, L2; fed L0, L1, L1g, L2 (16 arms, 185 records) | 0.0 | 1e-9 |

* **Fitted a and b.** At every level 2 arm, the a and b fitted at the
  deployment site equal those of the simulation (largest difference 0.0).
* **AIDS sites.** n_k = (951, 448, 312) and n_k^+ = (273, 127, 17). The trees
  per site are t = (56, 26, 18), and there was no out of bag fallback.
* **Diabetic sites.** n_k = (226, 234, 40, 420) and
  n_k^+ = (112, 122, 12, 242).
* **Run time.** Both configurations take about 65 s together on the Mac.

## 8. Tests

* **E1 environment.** `fed_rf_mk/tests/test_ladder.py` and
  `experiments/csbj/tests` give 71 passed and 1 skipped. The skipped one is the
  PySyft test.
* **PySyft environment.** `fed_rf_mk/tests` run from `fed_rf_mk`, as their
  relative paths require, gives 37 passed. This includes the existing
  `test_aggregation.py` (the old check of 6 + 4 trees still holds),
  `test_weights.py` and `test_remote_tasks.py`. New tests: largest remainder
  and seeded selection in `combine`, oob sampling without scores raising an
  error, and the counts and oob scores returned by `ml_experiment`.
* **Simulation unchanged.** `run_task.py`, `analyze.py`, `tables.py` and
  `figures.py` were not changed. The existing functions of `ladder.py` were
  not changed, and only `rf_tree_probs` was added. `numbers.json`, the tables
  and the figures were not regenerated and did not change.

## 9. What Section 2.7 says and the code does not do

The paper text was not available here. The local `fedrf_csbj_paper.pdf` and
`fedrf_csbj_paper.zip` (21 November) are an older version without the ladder,
in which Section 2.7 is "Federated Learning Protocol". The points below are
checked against Section 2.7 as described in `INSTRUCOES_E1.md` 9.1: the ladder
is implemented in fed_rf, and every site runs a PySyft DataSite that runs
submitted code only after the data owner approves it.

1. **Approval without review by default.** `FLServer` approves every request
   automatically unless it is started with `auto_accept=False`
   (`server.py`, `FLServer.__init__`; `datasites.py`,
   `check_and_approve_incoming_requests`). The code is then run without anyone
   reviewing it.
2. **Approval covers the code, not the parameters.** `dataParams` and
   `modelParams` are only typed (`orchestrator/clients.py`), and an approved
   function could be called again with other values.
3. **The mock is the real data by default** (`server.py`, `FLServer.__init__`;
   `datasites.py`, `create_syft_dataset`).
4. **No release per level.** One training round sends every site's full
   model, n_k and n_k^+ to the coordinator (`remote_tasks.py`,
   `ml_experiment`; `client.py`, `FLClient.run_model`). There is no option
   for a level 0 run in which only the source site releases its model and the
   other sites release nothing.
   * The deployment site receives the models of all K sites
     (`modelParams["ladder"]["models"]`), and evaluates every site as a source.
5. **The coordinator is not a DataSite.** Site weights, π, the tree allocation
   and the tree selection are computed in the requester's process, which sees
   every model and every count.
6. **Counts sent with every model.** n_k and n_k^+ leave every site with every
   model, even when only level 0 or level 3 without correction is used.
7. **Outputs that leave the deployment site.** The labeled records never leave
   it, but metrics computed from its labels (AUROC, Brier, MCC, and a and b of
   level 2) are returned to the requester. With `return_predictions=True`,
   per record outputs of the test part are returned too.
8. **The E1 numbers do not come from PySyft runs.** They come from the
   simulation (`experiments/csbj/run_task.py`), which calls the `ladder.py`
   functions directly. The PySyft path was shown to give the same predictions
   in the two configurations above (r = 0, f = 0). Heart, the other
   partitions, K values and draws, and the variants E5, E10, E12 and E13 were
   not run through PySyft. The PySyft path has no logistic regression learner
   (E13).
9. **fed_rf_mk must be installed at every DataSite.** The approved code imports
   `fed_rf_mk.ladder` there, and nothing in the request installs it.
10. **The analysis switch does not reach the approved code.**
    `FLServer(analysis_allowed=True)` does not reach the process where the
    approved code runs (section 6), so SHAP and PFI at the sites stay off in
    the local setup.

## 10. Changes to fed_rf in this step

* `fed_rf_mk/remote_tasks.py`
  * `ml_experiment` returns n_k, n_k^+ and, for oob sampling, the out of bag
    scores.
  * It takes `random_state` and the XGBoost `n_jobs` from `modelParams`.
  * It trains on every record when `train_size >= 1`, and casts the features to
    float64.
  * It imports `datasites` as `fed_rf_mk.datasites` first.
  * `evaluate_global_model` has the ladder branch.
* `fed_rf_mk/aggregation/models.py`
  * `merge_estimators` uses `allocate_trees` and `select_trees` with a seeded
    numpy generator, instead of `int(round(n_estimators*w))` trees per site
    with unseeded `random.sample`.
  * There are new `counts`, `site_weights` and `combine`.
* `fed_rf_mk/client.py`
  * There is a new `LADDER_DEFAULTS`.
  * `run_model` sends per site seeds, keeps the site order, skips sites that
    return an error, combines with `combine`, and puts `ladder` in
    `model_params`.
* `fed_rf_mk/ladder.py`: new function `rf_tree_probs`.
* **Behaviour changes for existing users.**
  * The weights given to `add_train_client` are used only with
    `site_weights="manual"`. The default is now `equal`.
  * The number of trees per site follows the largest remainder rule.
  * The selection is reproducible (seed 42 by default).
