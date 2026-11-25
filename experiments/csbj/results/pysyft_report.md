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
| counts only (level 1, sites other than the source) | each such training DataSite, as approved code | `fed_rf_mk/remote_tasks.py`, `site_counts` |
| what each site and the deployment site receive and release | the coordinator, bound in the code requests | `fed_rf_mk/release.py`, `site_plan`, `deployment_ladder`; `FLClient.send_request`, `FLClient.send_evaluation_request` |
| site weights, pooled prevalence, tree allocation and selection | the requester's Python process (the coordinator), not a DataSite | `fed_rf_mk/client.py`, `FLClient.run_model`; `fed_rf_mk/aggregation/models.py`, `ModelAggregator.combine`, `ModelAggregator.merge_estimators` |
| levels 0, 1, 2 and 3 at the deployment site | the evaluation DataSite, as approved code | `fed_rf_mk/remote_tasks.py`, `evaluate_global_model` (inner `_ladder_evaluate`) |

The approved code ran in the server process that `sy.orchestra.launch` starts
(`fed_rf_mk/datasites.py`, `spawn_server`). That process has a different pid
from the requester. The code imports `fed_rf_mk.ladder` inside that process. A
code request carries only the source of the function, so `fed_rf_mk` must be
installed at every DataSite. In the local test, the server process gets the
`sys.path` of the process that launched it.

## 3. What each training site sends, per level

`model_params["release_level"]` sets what leaves the training sites
(`fed_rf_mk/release.py`, `site_plan`; `fed_rf_mk/client.py`,
`FLClient.send_request`). The values are `0`, `1`, `2`, `3`, `3+1`, `3+2`
and `all`, with `all` as the default. Levels 0, 1 and 2 also need
`source_site`.

* **What the site sends.** The plan is put into the `modelParams` of each
  request as `release`, a dict of which items to send. The site returns only
  those items (`remote_tasks.py`, `ml_experiment`).
* **Counts only.** At level 1, the sites other than the source run
  `remote_tasks.site_counts`. It trains no model and returns only n_k and
  n_k^+.
* **Sites with no role.** A site with no role at a level gets no request.
* **No analysis outputs.** With a `release` dict, no SHAP or PFI output is
  computed or sent.
* **Level 3 and 3 with 2.** For these levels the site weights decide which
  counts leave. A site sends n_k with `size` weights, n_k^+ with `positives`
  weights, and no count with `equal` or `manual` weights.
* **Out of bag scores.** They leave only for Random Forest with
  `tree_sampling="oob"`, at 3, 3+1 and 3+2.

The table below is what `experiments/csbj/tests/test_pysyft_release.py`
observed. It used three training sites, the source `site1`, Random Forest,
`size` weights and `oob` sampling. The keys are those returned by each site
and stored in `FLClient.site_results`. The full record is in
`experiments/csbj/results/pysyft_release.json`.

| release_level | source site (`site1`) | other sites (`site0`, `site2`) |
|---|---|---|
| 0 | `ml_experiment`: model, model_type | no request |
| 1 | `ml_experiment`: model, model_type, n_train, n_pos | `site_counts`: n_train, n_pos |
| 2 | `ml_experiment`: model, model_type | no request |
| 3 | `ml_experiment`: model, model_type, n_train (size weights), oob_scores | the same |
| 3+1 | `ml_experiment`: model, model_type, n_train, n_pos, oob_scores | the same |
| 3+2 | `ml_experiment`: model, model_type, n_train (size weights), oob_scores | the same |
| all | `ml_experiment`, the full single round: model, model_type, n_train, n_pos, oob_scores, sample_size, analysis_status, n_base_estimators, n_incremental_estimators, train_size, test_size (plus shap_data and pfi_data when analysis is allowed) | the same |

* **Defaults of the training run.** With `train_size >= 1` (or missing),
  every record of the site trains its model (`ml_experiment`, `preprocess`).
* **Model seed.** The local seed of the model is
  `modelParams["random_state"]` (default 42). `FLClient` sets it per site from
  `site_seeds`. `site_seeds` itself stays at the coordinator.

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
* **Levels 0, 1 and 2.** There is no combination (`FLClient._source_ladder`).
  The ladder dict holds the model of the source and its index among the
  training sites. For level 1 it also holds π (`ladder.pooled_prevalence`,
  from the counts of every site) and π_k of the source.
* **What goes to the deployment site.** `release.deployment_ladder` keeps only
  the keys that the level needs, and the evaluation request binds them. The
  keys below were observed in `test_pysyft_release.py` (Random Forest; the
  last five keys of the level 3 list are specific to Random Forest).

  | release_level | keys of `modelParams["ladder"]` | models |
  |---|---|---|
  | 0, 2 | model_type, sites, sources, models | 1 (source) |
  | 1 | model_type, sites, sources, models, pi, pi_k | 1 (source) |
  | 3, 3+2 | model_type, sites, sources, models, weights, weights_kind, sampling, seed, T_k, t, selections, oob_fallback | K |
  | 3+1 | as 3, plus pi, pi_k | K |
  | all | as 3+1, plus n, n_pos | K |

  * For every level except `all`, the other keys of the evaluation
    `modelParams` are only `model_type`, `release_level`, `calibration` and
    `return_predictions`.
  * For `all`, the deployment site gets the full `model_params`, as before,
    without `site_seeds` and `approval_timeout`. That includes `model` and
    `ensemble_members`.
* **π_k.** It is computed at the coordinator as n_k^+/n_k (int64 division,
  as in the simulation) and sent as `pi_k`.
* **SHAP and PFI.** They are averaged with the same site weights (only `all` sends them).

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
* **Arms per level.** The deployment site computes only the arms that its
  `release_level` allows (`evaluate_global_model`, `_ladder_evaluate`). It
  checks that what it needs was received.

  | release_level | arms |
  |---|---|
  | 0 | src_L0 (source) |
  | 1 | src_L0, src_L1 (source) |
  | 2 | src_L0, src_L2 (source) |
  | 3 | src_L0 of every site, fed_L0 |
  | 3+1 | src_L0, src_L1 of every site, fed_L0, fed_L1, fed_L1g |
  | 3+2 | src_L0, src_L2 of every site, fed_L0, fed_L2 |
  | all | every arm |

  `test_pysyft_release.py` observed exactly these arms for every level, with
  source `site1`, so for example `src1_L0` at level 0. MCC at π is None when π
  was not released (levels 0, 2, 3 and 3+2).
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
| `release_level` | `all` |
| `source_site` | None (needed for 0, 1, 2) |
| `approval_timeout` | 300 s |

## 6. How code requests are approved

* **Requests with bound values.**
  * `DataSiteManager._send_code_request` (`fed_rf_mk/orchestrator/clients.py`)
    wraps the site function in `sy.syft_function`, with
    `MixedInputPolicy(client, data=<asset>, dataParams=sy.Constant(val=...), modelParams=sy.Constant(val=...))`.
    It is used by `send_training_requests` and `send_evaluation_requests`.
  * Each request is put in a `sy.Project` and sent.
  * PySyft 0.9.5 can bind values this way. `sy.Constant` is
    `CreatePolicyRuleConstant`, which becomes the `Constant` rule of
    `syft/service/policy/policy.py`, and `Constant.transform_kwarg` returns the
    bound value.
  * The approved code therefore runs with the bound values. A call that passes
    other values gets the bound ones instead, so other values need a new
    request and a new approval. This was observed twice: in a scratch check,
    and in `test_mock_accounts_binding_analysis`, where `ml_experiment` was
    called with `train_size=0.5` and still trained on the bound
    `train_size=0.7`.
  * `FLClient.run_model` and `FLClient.run_evaluate` call the approved
    functions with `data` only.
* **Order of the requests.**
  * `FLClient.send_request` sends only the training requests.
  * The evaluation request carries the released models, so it can only be sent
    after training. It is sent by `FLClient.send_evaluation_request`, or by
    `run_evaluate` when that has not happened yet.
  * `run_model` and `run_evaluate` wait up to `approval_timeout` seconds
    (default 300) for the approval (`DataSiteManager.wait_for_approval`). They
    stop when a request is rejected.
  * The old fallback of `run_evaluate`, which built and called an unapproved
    function, was removed.
* **Who approves.** The data owner, logged in as the DataSite admin, approves.
  * `FLServer.__init__` now has `auto_accept=False` as its default.
  * The manual flow is in `fed_rf_mk/server.py`:
    * `FLServer.list_pending_requests` returns the index and function name of
      each pending request. It now reads `client.requests`, since `client.code`
      has no `len` in 0.9.5.
    * `FLServer.inspect_request(i)` returns the code.
    * `FLServer.request_parameters(i)` returns the bound `dataParams` and
      `modelParams`, a new method.
    * `FLServer.approve_request(i)` approves.
  * `test_pysyft_release.py` uses this manual flow for every release level. It
    checks that the bound values equal those the client sent.
  * With `auto_accept=True`, passed explicitly, the thread
    `check_and_approve_incoming_requests` (`datasites.py`) still approves every
    request once a second without looking at it.
  * `auto_accept=True` is passed explicitly in the equivalence test, in the
    README quickstart, and in the example notebook (datasite 2 only, as
    before).
* **Mock data seen by the requester.**
  * `FLServer` no longer sets `mock_path = data_path`.
  * Without a mock file, `create_syft_dataset` (`datasites.py`) uploads
    `datasets.generate_mock(data)` and logs that a synthetic mock was used.
  * The synthetic mock has the columns and dtypes of the data and 100 rows. The
    row count is fixed, so it does not reveal the size of the site. Values are
    drawn independently for each column with seed 0. Numeric columns are
    uniform between their minimum and maximum (integers inclusive). Other
    columns, booleans included, are uniform over their observed values.
  * Through the minimum, the maximum and the observed categories, the mock
    still reveals the range of each column and the set of categories.
  * The dead code after `return data` in `generate_mock` was removed.
  * Logged in as the data scientist, the mock had the columns and dtypes of the
    data and shared no row with it.
* **Accounts.** `spawn_server` does the following.
  * The data owner gives the new admin password and the data scientist
    password, as arguments of `spawn_server` or `FLServer`, or in
    `FEDRF_ADMIN_PASSWORD` and `FEDRF_DS_PASSWORD`. The email comes from
    `FEDRF_DS_EMAIL` and defaults to `fedlearning@rf.com`.
  * Without a password it stops with an error, and the PySyft default
    `changethis` is refused as the new admin password.
  * It logs in with the default admin password and changes it
    (`client.account.set_password(..., confirm=False)`). A later login with
    `changethis` failed in the test.
  * Guest signup is disabled unless `allow_guest_signup=True` or
    `FEDRF_ALLOW_GUEST_SIGNUP=1`. `settings.signup_enabled` was False in the
    test.
  * No password is written in the code. The tests make random ones with
    `secrets.token_urlsafe`, and the notebook asks for them with `getpass`.
* **Analysis policy.**
  * `datasites.set_analysis_allowed`, which `FLServer.start` calls with
    `analysis_allowed`, now also writes `FEDRF_ANALYSIS_ALLOWED` to the
    environment.
  * `is_analysis_allowed` reads that variable first.
  * The DataSite process is started after the variable is set and inherits it,
    so the approved code sees the policy of its DataSite.
  * In the test, `ml_experiment` returned `analysis_status="enabled"` with
    `pfi_data` under `analysis_allowed=True`, and `"disabled_by_silo_policy"`
    with no `pfi_data` under `False`.
  * Limits:
    * A change made after the DataSite started (`FLServer.set_analysis_policy`)
      does not reach it.
    * The variable is per process, so DataSites started from one process take
      the value set last before each launch.

## 7. Equivalence with the simulation

The test is `experiments/csbj/tests/test_pysyft_equivalence.py`. The full
output is in `experiments/csbj/results/pysyft_equivalence.json`.

* **Data and partition.** Both configurations use r = 0, f = 0 and the
  partition and seeds of `experiments/csbj/run_task.py`.
  * Every site and the deployment site get a CSV with their records, in the
    order of the simulation. The test checks that the CSV round trip is exact.
  * Each one is served by an `FLServer` with `auto_accept=True` passed explicitly,
    random passwords and the synthetic mock.
  * `FLClient` runs `send_request`, `run_model` and `run_evaluate`, with
    `release_level="all"` and the values bound in the requests.
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
* **Rerun.** The test was rerun after the changes of 1b, with the same result:
  largest difference 0.0 in every arm, and in a and b.
* **Run time.** Both configurations take about 60 s together on the Mac.

## 8. Tests

* **E1 environment.** `fed_rf_mk/tests/test_ladder.py` and
  `experiments/csbj/tests` give 71 passed and 2 skipped. The skipped ones are
  the two PySyft tests.
* **PySyft environment, `fed_rf_mk/tests`.** Run from `fed_rf_mk`, as their
  relative paths require: 40 passed. They include `test_aggregation.py`,
  `test_weights.py`, `test_remote_tasks.py` and the new `test_datasets.py`
  (synthetic mock, release plan, deployment keys).
* **PySyft environment, `experiments/csbj/tests`.** 54 passed, in about 5
  minutes:
  * the E1 tests;
  * `test_pysyft_equivalence.py` (2 configurations);
  * `test_pysyft_release.py`: 7 release levels with manual approval, plus mock,
    accounts, binding and analysis switch.
* **Simulation unchanged.**
  * `run_task.py`, `analyze.py`, `tables.py` and `figures.py` were not changed.
  * The functions of `ladder.py` were not changed in 1b.
  * `numbers.json`, the tables and the figures were not regenerated and did not
    change.

## 9. What Section 2.7 of the paper and Section S3 of the SI say and the code does not do

* **Claims compared with the code.** Section 2.7 of the paper and Section S3
  of the Supplementary Information claim two things:
  * the data owners approve the code, and the values of the parameters, that
    run on their data;
  * the disclosure ladder, including what each level releases, is implemented
    in fed_rf.
* **Closed in this step.** Approval is manual by default, the mock is never
  the real data, approval covers the parameter values, there is a release per
  level, no password is written in the code, and the analysis switch reaches
  the DataSite.

What the code still does not do, or does only in part:

1. **Approval without review is still possible.** `auto_accept=True` still
   approves every request without review. It must be passed explicitly. The
   equivalence test uses it.
2. **The default is still the single round.** `release_level="all"` sends
   every site's full model, counts, out of bag scores (for oob sampling) and
   training parameters. It also sends the analysis outputs when they are
   allowed. The deployment site then receives the full `model_params`. The
   other values release only what the level needs (section 3).
3. **Weights reveal relative sizes.** At 3 and 3+2 with `size` or `positives`
   weights, the counts that the weights need leave every site. With any
   weights, the deployment site receives the normalized weights, which give
   the relative site sizes for `size` weights.
4. **The coordinator is not a DataSite.** Site weights, π, the tree allocation
   and the tree selection are computed in the requester's process, which sees
   every released model and count.
5. **The deployment site sees every source at level 3.** At 3, 3+1 and 3+2 it
   receives every model, and it also evaluates each model as a source
   (`src<k>_L0`, and L1 or L2 where the level allows it).
6. **Outputs that leave the deployment site.** The labeled records never leave
   it, but metrics computed from its labels are returned to the requester:
   AUROC, AUPRC, Brier, MCC, and a and b of level 2. With
   `return_predictions=True`, per record outputs of the test part are returned
   too.
7. **The E1 numbers do not come from PySyft runs.** They come from the
   simulation (`experiments/csbj/run_task.py`). The PySyft path gives the same
   predictions in the two configurations of section 7. Heart, the other
   partitions, K values and draws, E5, E10, E12 and E13 were not run through
   PySyft, and there is no logistic regression learner on the PySyft path.
8. **fed_rf_mk must be installed at every DataSite.** The approved code imports
   `fed_rf_mk.ladder` there, and nothing in the request installs it.
9. **Limits of the analysis switch.** See section 6: a change made after
   launch is not seen by the DataSite, and the variable is per process.
10. **The synthetic mock reveals column ranges and categories.** It shows the
    range of every numeric column and the set of values of every other column,
    the label included.
11. **Approval binds values, not use.** It covers the code and the bound
    values. A function approved once ran a second time in the check, even
    though the syft output policy is `OutputPolicyExecuteOnce`. The second run
    used the same bound values.

## 10. Changes to fed_rf

### 9.1

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

### 1b

* `fed_rf_mk/server.py`
  * `FLServer` has `auto_accept=False` by default.
  * It no longer sets `mock_path = data_path`.
  * It passes the account settings to `spawn_server`.
  * `list_pending_requests` works with syft 0.9.5 and returns the pending
    requests.
  * There is a new `request_parameters`.
* `fed_rf_mk/datasites.py`
  * Synthetic mock when there is no mock file.
  * Admin password changed at start, guest signup off by default, data
    scientist password given by the data owner (arguments or environment).
  * The analysis policy goes through `FEDRF_ANALYSIS_ALLOWED`.
* `fed_rf_mk/datasets.py`: `generate_mock` builds the synthetic mock, and the
  dead code is gone.
* `fed_rf_mk/release.py` (new): `site_plan`, `released_keys` and
  `deployment_ladder`.
* `fed_rf_mk/remote_tasks.py`
  * `ml_experiment` honours `release`.
  * There is a new `site_counts`.
  * The ladder branch of `evaluate_global_model` computes only the arms of its
    `release_level`, and takes `pi_k` from the coordinator.
* `fed_rf_mk/orchestrator/clients.py`: requests bind `dataParams` and
  `modelParams` with `sy.Constant`, and there is a new `wait_for_approval`.
* `fed_rf_mk/client.py`
  * `send_request` follows the release plan, and there is a new
    `send_evaluation_request`.
  * `run_model` waits for approvals and handles levels 0 to 2.
  * `run_evaluate` sends the evaluation request after training and has no
    unapproved fallback.
  * `site_results`, `site_params_sent` and `eval_params_sent` keep what was
    sent and received.
* `fed_rf_mk/aggregation/models.py`: partial counts for `size` and
  `positives` weights, and `pi_k` and `sources` in the ladder dict.
* `README.md` quickstart and `examples/aids_clinical_dataset.ipynb`
  * Passwords come from the environment (`getpass` in the notebook).
  * `send_request` is called explicitly.
  * The evaluation request is approved after training in the notebook.
* **Behaviour changes for existing users.**
  * `FLServer` needs passwords and approves by hand by default.
  * The mock is synthetic.
  * The evaluation request is sent after training, and `run_model` and
    `run_evaluate` wait for approvals.
  * The approved code runs only with the values bound at request time.

