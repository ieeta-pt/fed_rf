# Changelog

## 1.1.0

Disclosure ladder of the CSBJ paper, and changes to the PySyft path.

### Added

* `fed_rf_mk/ladder.py`: the disclosure ladder.
  * Pooled prevalence and site weights (`equal`, `size`, `positives`).
  * Level 1 shift (`shift_prob`, `shift_logit`) and level 2 recalibration.
  * Largest remainder tree allocation and seeded tree selection, uniform or by
    out of bag score.
  * Federated Random Forest and XGBoost, with and without the level 1
    correction.
  * Out of bag scores and single record leaf statistics.

  It needs only numpy, scipy, scikit-learn and xgboost, not PySyft.
* `fed_rf_mk/release.py` and `model_params["release_level"]` (`0`, `1`, `2`,
  `3`, `3+1`, `3+2`, `all`) with `source_site`.
  * For every value other than `all`, only what the level needs leaves the
    training sites.
  * The deployment site receives only what was released, and computes only
    those levels.
  * New remote function `site_counts`, which returns n_k and n_k^+ only.
* **Training sites.** `ml_experiment` returns n_k and n_k^+, and, for out of
  bag sampling, the out of bag score of each tree.
* **Deployment site.** `evaluate_global_model` computes the levels of the
  ladder at the deployment site (levels 0 to 3, 3 with 1, 3 with 2). Labels
  never leave the site.
* **New `model_params` options.** `site_weights`, `tree_sampling`, `seed`,
  `site_seeds`, `calibration`, `return_predictions`, `release_level`,
  `source_site` and `approval_timeout`.
* **New methods.**
  * `FLClient.send_evaluation_request`.
  * `FLServer.request_parameters`, which shows the values bound in a request.
* **Experiments.** `experiments/csbj`: the simulation, analysis, tables,
  figures and results of the CSBJ paper. It includes the tests that the
  PySyft path gives the predictions of the simulation and releases only what
  each level needs.

### Changed (behaviour)

* **Default site weights.** The default is now `equal`. The weights given to
  `FLClient.add_train_client` are used only with `site_weights="manual"`.
  Before, they were always used.
* **Tree allocation.** Random Forest trees per site follow the largest
  remainder rule (`ladder.allocate_trees`). Before, each site gave
  `int(round(n_estimators * w))` trees.
* **Seeded selection.** Trees are chosen with a seeded numpy generator
  (`seed`, default 42), so the merged forest is reproducible. Before, they
  were chosen with unseeded `random.sample`. Sites are taken in the order they
  were added, not in the order they finished.
* **Manual approval.** `FLServer` approves code requests by hand by default
  (`auto_accept=False`). Pass `auto_accept=True` to approve automatically.
* **Bound values.** The values of `dataParams` and `modelParams` are bound in
  each code request, so an approval covers these values. A call with other
  values runs with the bound ones.
* **Evaluation request after training.** `send_request` sends only the
  training requests. The evaluation request carries the released models and is
  sent after training (`send_evaluation_request`, or by `run_evaluate`).
  * `run_model` and `run_evaluate` wait for approvals up to `approval_timeout`.
  * The unapproved fallback of `run_evaluate` was removed.
* **Synthetic mock.** The real data is never uploaded as the mock. Without a
  `mock_path`, each DataSite builds a synthetic mock with 100 rows, the columns
  and dtypes of the data, and values drawn independently per column with a
  fixed seed.
* **Accounts.**
  * `spawn_server` and `FLServer` replace the PySyft default admin password at
    start, with a password given by the data owner.
  * They create the data scientist account with a password given by the data
    owner.
  * The values come from arguments, or from `FEDRF_ADMIN_PASSWORD`,
    `FEDRF_DS_PASSWORD` and `FEDRF_DS_EMAIL`.
  * Guest signup is disabled unless asked for.
* **Analysis switch.** `analysis_allowed` of `FLServer` now reaches the
  DataSite process, through `FEDRF_ANALYSIS_ALLOWED`. It must be set before
  the DataSite starts.
* **Training on every record.** `train_size >= 1` (or missing) trains on every
  record of a site, and features are cast to float64.
