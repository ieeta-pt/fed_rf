# numbers

Every value is computed from the run files by `analyze.py report`.
Intervals are 95% percentile bootstrap intervals over draws (B = 2000).
Learners rf and xgb only (E13, logistic regression, is not reported).

| key | value | 95% CI | J | how |
|---|---|---|---|---|
| `design.n_draws.aids` | 50 |  |  | Distinct (rep, fold) draws in the main grid. |
| `design.n_repetitions.aids` | 10 |  |  | Repetitions of 5 fold CV in the main grid. |
| `design.n_draws.diabetic` | 50 |  |  | Distinct (rep, fold) draws in the main grid. |
| `design.n_repetitions.diabetic` | 10 |  |  | Repetitions of 5 fold CV in the main grid. |
| `design.n_draws.heart` | 50 |  |  | Distinct (rep, fold) draws in the main grid. |
| `design.n_repetitions.heart` | 10 |  |  | Repetitions of 5 fold CV in the main grid. |
| `design.n_folds` | 5 |  |  | Folds per repetition. |
| `design.n_draws` | 50 |  |  | Same for every dataset (main grid). |
| `design.n_repetitions` | 10 |  |  | Same for every dataset (main grid). |
| `design.regularized.n_draws` | 20 |  |  | Same for every dataset (regularized grid, common draws). |
| `design.regularized.n_repetitions` | 4 |  |  | Same for every dataset (regularized grid, common draws). |
| `design.regularized.n_draws.aids` | 20 |  |  | Draws of the regularized grid shared with the main grid (common draws). |
| `design.regularized.n_repetitions.aids` | 4 |  |  | Repetitions in the common draws of the regularized grid. |
| `design.regularized.n_draws.diabetic` | 20 |  |  | Draws of the regularized grid shared with the main grid (common draws). |
| `design.regularized.n_repetitions.diabetic` | 4 |  |  | Repetitions in the common draws of the regularized grid. |
| `design.regularized.n_draws.heart` | 20 |  |  | Draws of the regularized grid shared with the main grid (common draws). |
| `design.regularized.n_repetitions.heart` | 4 |  |  | Repetitions in the common draws of the regularized grid. |
| `design.min_per_class` | 5 |  |  | Records of each class given to every site. |
| `design.dirichlet_alpha` | [0.1000, 0.5000] |  |  | Dirichlet concentrations of the label skewed partitions. |
| `design.K` | [2, 3, 4, 5, 6, 7, 8, 9, 10] |  |  | Numbers of sites. |
| `data.aids.n_rows` | 2139 |  |  | From the data loader used by the runs. |
| `data.aids.n_features` | 19 |  |  | From the data loader used by the runs. |
| `data.aids.n_positives` | 521 |  |  | From the data loader used by the runs. |
| `data.diabetic.n_rows` | 1151 |  |  | From the data loader used by the runs. |
| `data.diabetic.n_features` | 19 |  |  | From the data loader used by the runs. |
| `data.diabetic.n_positives` | 611 |  |  | From the data loader used by the runs. |
| `data.heart.n_rows` | 246022 |  |  | From the data loader used by the runs. |
| `data.heart.n_features` | 38 |  |  | From the data loader used by the runs. |
| `data.heart.n_positives` | 13435 |  |  | From the data loader used by the runs. |
| `hyper.rf.n_estimators` | 100 |  |  | get_params() written by the runs (hyperparameters.json). |
| `hyper.xgb.n_estimators` | 100 |  |  | get_params() written by the runs (hyperparameters.json). |
| `hyper.xgb.max_depth` | 6 |  |  | get_params() written by the runs (hyperparameters.json). |
| `hyper.xgb.learning_rate` | 0.3000 |  |  | get_params() written by the runs (hyperparameters.json). |
| `runs.error_files` | 0 |  |  | Error files in the run folders (failed cells). |
| `runs.missing_files` | 0 |  |  | Expected outputs absent. |
| `cal.mean_size.aids` | 85.0000 |  |  | Mean size of the calibration part (20% of the deployment site). |
| `cal.mean_size.diabetic` | 46.0000 |  |  | Mean size of the calibration part (20% of the deployment site). |
| `cal.mean_size.heart` | 9840 |  |  | Mean size of the calibration part (20% of the deployment site). |
| `r.brier.L1.aids.rf.equal` | 0.0002 | [0.0001, 0.0003] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.aids.rf.equal` | 0.1772 | [0.0471, 0.2823] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.aids.rf.equal` | 0.1615 | [0.0889, 0.3805] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.aids.rf.equal` | 2.7925 | [1.8536, 6.9712] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L1.aids.rf.equal` | 0.0000 | [0.0000, 0.0000] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.aids.rf.equal` | -0.3815 | [-0.6407, -0.1939] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.aids.rf.dir0.5` | 0.8010 | [0.7902, 0.8108] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.aids.rf.dir0.5` | 0.9411 | [0.9335, 0.9484] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.aids.rf.dir0.5` | 0.6166 | [0.5502, 0.6968] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.aids.rf.dir0.5` | -2.9292 | [-17.2704, 10.2058] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: no. |
| `r.mcc_pi.L1.aids.rf.dir0.5` | 0.5807 | [0.5497, 0.6113] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.aids.rf.dir0.5` | 0.3022 | [0.2372, 0.3580] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.aids.rf.dir0.1` | 0.7321 | [0.7226, 0.7413] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.aids.rf.dir0.1` | 0.9763 | [0.9730, 0.9795] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.aids.rf.dir0.1` | 0.5472 | [0.4777, 0.6199] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.aids.rf.dir0.1` | -7.1654 | [-89.8753, 74.1400] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: no. |
| `r.mcc_pi.L1.aids.rf.dir0.1` | 0.7158 | [0.6704, 0.7670] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.aids.rf.dir0.1` | 0.3577 | [0.2972, 0.4119] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.aids.xgb.equal` | 0.0000 | [-0.0000, 0.0001] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.aids.xgb.equal` | 0.8307 | [0.8092, 0.8509] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.aids.xgb.equal` | 0.0020 | [-0.0007, 0.0046] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.aids.xgb.equal` | 10.0891 | [-82.7451, 87.1695] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: no. |
| `r.mcc_pi.L1.aids.xgb.equal` | -0.0006 | [-0.0019, 0.0007] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.aids.xgb.equal` | -0.2650 | [-0.3660, -0.1796] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.aids.xgb.dir0.5` | 0.4789 | [0.4698, 0.4873] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.aids.xgb.dir0.5` | 0.9482 | [0.9427, 0.9537] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.aids.xgb.dir0.5` | 0.2539 | [0.2239, 0.2866] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.aids.xgb.dir0.5` | -11.6436 | [-107.3356, 86.8012] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: no. |
| `r.mcc_pi.L1.aids.xgb.dir0.5` | 0.1778 | [0.1608, 0.1946] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.aids.xgb.dir0.5` | 0.0556 | [0.0033, 0.1035] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.aids.xgb.dir0.1` | 0.5926 | [0.5859, 0.5998] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.aids.xgb.dir0.1` | 0.9737 | [0.9708, 0.9767] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.aids.xgb.dir0.1` | 0.3934 | [0.3391, 0.4571] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.aids.xgb.dir0.1` | -5.2118 | [-43.9021, 38.0989] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: no. |
| `r.mcc_pi.L1.aids.xgb.dir0.1` | 0.2567 | [0.2340, 0.2793] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.aids.xgb.dir0.1` | 0.1987 | [0.1476, 0.2446] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.diabetic.rf.equal` | 0.0002 | [0.0001, 0.0003] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.diabetic.rf.equal` | -0.2126 | [-0.4175, -0.0861] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.diabetic.rf.equal` | -0.0242 | [-0.0330, -0.0162] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.diabetic.rf.equal` | -0.2464 | [-0.4760, -0.1086] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L1.diabetic.rf.equal` | -0.0089 | [-0.0152, -0.0035] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.diabetic.rf.equal` | -0.1142 | [-0.2517, -0.0302] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.diabetic.rf.dir0.5` | 0.6223 | [0.6120, 0.6323] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.diabetic.rf.dir0.5` | 0.8092 | [0.7936, 0.8248] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.diabetic.rf.dir0.5` | 0.3697 | [0.3436, 0.3960] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.diabetic.rf.dir0.5` | 0.1628 | [0.1011, 0.2161] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L1.diabetic.rf.dir0.5` | 0.3488 | [0.3238, 0.3743] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.diabetic.rf.dir0.5` | 0.2113 | [0.1633, 0.2547] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.diabetic.rf.dir0.1` | 0.5244 | [0.5141, 0.5348] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.diabetic.rf.dir0.1` | 0.9108 | [0.8997, 0.9235] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.diabetic.rf.dir0.1` | 0.3935 | [0.3659, 0.4193] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.diabetic.rf.dir0.1` | 0.2547 | [0.1984, 0.3075] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L1.diabetic.rf.dir0.1` | 0.3859 | [0.3546, 0.4154] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.diabetic.rf.dir0.1` | 0.2737 | [0.2287, 0.3211] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.diabetic.xgb.equal` | 0.0000 | [-0.0000, 0.0000] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.diabetic.xgb.equal` | 0.6532 | [0.6243, 0.6830] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.diabetic.xgb.equal` | 0.0003 | [-0.0011, 0.0017] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.diabetic.xgb.equal` | -0.1385 | [-0.2153, -0.0771] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L1.diabetic.xgb.equal` | 0.0003 | [-0.0009, 0.0014] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.diabetic.xgb.equal` | -0.0350 | [-0.0836, 0.0070] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.diabetic.xgb.dir0.5` | 0.3443 | [0.3351, 0.3531] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.diabetic.xgb.dir0.5` | 0.8288 | [0.8151, 0.8429] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.diabetic.xgb.dir0.5` | 0.1746 | [0.1555, 0.1947] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.diabetic.xgb.dir0.5` | 0.0503 | [0.0108, 0.0880] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L1.diabetic.xgb.dir0.5` | 0.1678 | [0.1499, 0.1863] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.diabetic.xgb.dir0.5` | 0.1079 | [0.0739, 0.1397] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.diabetic.xgb.dir0.1` | 0.4113 | [0.4009, 0.4214] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.diabetic.xgb.dir0.1` | 0.8757 | [0.8649, 0.8867] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.diabetic.xgb.dir0.1` | 0.2186 | [0.1984, 0.2401] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.diabetic.xgb.dir0.1` | 0.1141 | [0.0742, 0.1517] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L1.diabetic.xgb.dir0.1` | 0.2183 | [0.1992, 0.2386] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.diabetic.xgb.dir0.1` | 0.1388 | [0.1080, 0.1713] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.heart.rf.equal` | 0.0000 | [-0.0000, 0.0000] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.heart.rf.equal` | 0.3022 | [0.2833, 0.3195] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.heart.rf.equal` | -0.2337 | [-0.2616, -0.2110] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.heart.rf.equal` | 1.2170 | [1.1326, 1.3261] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L1.heart.rf.equal` | 0.0000 | [0.0000, 0.0000] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.heart.rf.equal` | 0.4316 | [0.3771, 0.4810] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.heart.rf.dir0.5` | 0.9095 | [0.8894, 0.9277] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.heart.rf.dir0.5` | 0.9790 | [0.9768, 0.9809] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.heart.rf.dir0.5` | 0.7021 | [0.6402, 0.7648] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.heart.rf.dir0.5` | 1.1532 | [1.0837, 1.2383] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L1.heart.rf.dir0.5` | 0.8659 | [0.8284, 0.9044] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.heart.rf.dir0.5` | 0.6712 | [0.6575, 0.6849] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.heart.rf.dir0.1` | 0.7033 | [0.6827, 0.7237] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.heart.rf.dir0.1` | 0.9926 | [0.9921, 0.9930] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.heart.rf.dir0.1` | 0.7886 | [-2.1735, 3.8925] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: no. |
| `r.mcc_05.L2.heart.rf.dir0.1` | 1.9183 | [1.6612, 2.3473] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L1.heart.rf.dir0.1` | 1.5948 | [1.4511, 1.7685] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.heart.rf.dir0.1` | 0.5822 | [0.5725, 0.5914] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.heart.xgb.equal` | -0.0000 | [-0.0000, 0.0000] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.heart.xgb.equal` | 0.5162 | [0.5096, 0.5231] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.heart.xgb.equal` | 0.0001 | [-0.0003, 0.0004] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.heart.xgb.equal` | 2.2613 | [2.0916, 2.4554] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L1.heart.xgb.equal` | 0.0000 | [-0.0001, 0.0001] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.heart.xgb.equal` | -0.5087 | [-0.5455, -0.4720] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.heart.xgb.dir0.5` | 0.8098 | [0.8019, 0.8181] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.heart.xgb.dir0.5` | 0.9755 | [0.9730, 0.9776] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.heart.xgb.dir0.5` | -0.2995 | [-0.4689, -0.1898] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.heart.xgb.dir0.5` | 2.2063 | [2.0535, 2.3884] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L1.heart.xgb.dir0.5` | 0.4601 | [0.4522, 0.4679] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.heart.xgb.dir0.5` | 0.5447 | [0.5357, 0.5538] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.heart.xgb.dir0.1` | 0.8259 | [0.8196, 0.8325] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L2.heart.xgb.dir0.1` | 0.9919 | [0.9915, 0.9923] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L1.heart.xgb.dir0.1` | -2.0183 | [-4.0285, -1.2374] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_05.L2.heart.xgb.dir0.1` | 2.0417 | [1.8846, 2.2287] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L1.heart.xgb.dir0.1` | 0.4560 | [0.4460, 0.4660] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.mcc_pi.L2.heart.xgb.dir0.1` | 0.5402 | [0.5316, 0.5482] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Reportable: yes. |
| `r.brier.L1.equal.range` | [-0.0000, 0.0002] |  |  | Minimum and maximum over the dataset and learner pairs present. |
| `r.brier.L1.dir0.5.range` | [0.3443, 0.9095] |  |  | Minimum and maximum over the dataset and learner pairs present. |
| `r.brier.L1.dir0.1.range` | [0.4113, 0.8259] |  |  | Minimum and maximum over the dataset and learner pairs present. |
| `r.brier.L2.equal.range` | [-0.2126, 0.8307] |  |  | Minimum and maximum over the dataset and learner pairs present. |
| `r.brier.L2.dir0.5.range` | [0.8092, 0.9790] |  |  | Minimum and maximum over the dataset and learner pairs present. |
| `r.brier.L2.dir0.1.range` | [0.8757, 0.9926] |  |  | Minimum and maximum over the dataset and learner pairs present. |
| `r.mcc_pi.L1.equal.range` | [-0.0089, 0.0003] |  |  | Minimum and maximum over the dataset and learner pairs present. |
| `r.mcc_pi.L1.dir0.5.range` | [0.1678, 0.8659] |  |  | Minimum and maximum over the dataset and learner pairs present. |
| `r.mcc_pi.L1.dir0.1.range` | [0.2183, 1.5948] |  |  | Minimum and maximum over the dataset and learner pairs present. |
| `r.mcc_pi.L2.equal.range` | [-0.5087, 0.4316] |  |  | Minimum and maximum over the dataset and learner pairs present. |
| `r.mcc_pi.L2.dir0.5.range` | [0.0556, 0.6712] |  |  | Minimum and maximum over the dataset and learner pairs present. |
| `r.mcc_pi.L2.dir0.1.range` | [0.1388, 0.5822] |  |  | Minimum and maximum over the dataset and learner pairs present. |
| `diff.brier.L1.aids.rf.equal` | -0.0000 | [-0.0000, -0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.aids.rf.equal` | -0.0011 | [-0.0019, -0.0003] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.aids.rf.equal` | -0.0084 | [-0.0086, -0.0082] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.aids.rf.equal` | -0.0084 | [-0.0086, -0.0082] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.aids.rf.equal` | -0.0061 | [-0.0070, -0.0052] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.aids.rf.equal` | -0.0042 | [-0.0050, -0.0034] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.aids.rf.equal` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.aids.rf.equal` | -0.0101 | [-0.0160, -0.0052] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.aids.rf.equal` | 0.0340 | [0.0318, 0.0362] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.aids.rf.equal` | 0.0340 | [0.0318, 0.0362] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.aids.rf.equal` | 0.0329 | [0.0302, 0.0354] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.aids.rf.equal` | 0.0208 | [0.0171, 0.0246] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.aids.rf.dir0.5` | -0.0731 | [-0.0773, -0.0688] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.aids.rf.dir0.5` | -0.0877 | [-0.0926, -0.0826] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.aids.rf.dir0.5` | -0.0780 | [-0.0814, -0.0745] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.aids.rf.dir0.5` | -0.0913 | [-0.0958, -0.0865] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.aids.rf.dir0.5` | -0.0932 | [-0.0980, -0.0880] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.aids.rf.dir0.5` | -0.0935 | [-0.0987, -0.0881] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.aids.rf.dir0.5` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.aids.rf.dir0.5` | -0.0114 | [-0.0156, -0.0078] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.aids.rf.dir0.5` | 0.0458 | [0.0428, 0.0489] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.aids.rf.dir0.5` | 0.0369 | [0.0344, 0.0394] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.aids.rf.dir0.5` | 0.0426 | [0.0389, 0.0462] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.aids.rf.dir0.5` | 0.0505 | [0.0452, 0.0558] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.aids.rf.dir0.1` | -0.1075 | [-0.1114, -0.1036] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.aids.rf.dir0.1` | -0.1536 | [-0.1591, -0.1479] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.aids.rf.dir0.1` | -0.1128 | [-0.1163, -0.1091] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.aids.rf.dir0.1` | -0.1469 | [-0.1513, -0.1423] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.aids.rf.dir0.1` | -0.1573 | [-0.1628, -0.1517] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.aids.rf.dir0.1` | -0.1619 | [-0.1676, -0.1561] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.aids.rf.dir0.1` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.aids.rf.dir0.1` | -0.0118 | [-0.0158, -0.0080] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.aids.rf.dir0.1` | 0.0474 | [0.0445, 0.0505] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.aids.rf.dir0.1` | 0.0382 | [0.0353, 0.0412] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.aids.rf.dir0.1` | 0.0315 | [0.0248, 0.0373] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.aids.rf.dir0.1` | 0.0865 | [0.0797, 0.0934] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.aids.xgb.equal` | -0.0000 | [-0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.aids.xgb.equal` | -0.0403 | [-0.0418, -0.0387] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.aids.xgb.equal` | -0.0324 | [-0.0331, -0.0316] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.aids.xgb.equal` | -0.0324 | [-0.0331, -0.0316] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.aids.xgb.equal` | -0.0485 | [-0.0495, -0.0474] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.aids.xgb.equal` | -0.0151 | [-0.0169, -0.0132] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.aids.xgb.equal` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.aids.xgb.equal` | -0.0123 | [-0.0174, -0.0082] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.aids.xgb.equal` | 0.0588 | [0.0559, 0.0620] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.aids.xgb.equal` | 0.0588 | [0.0559, 0.0620] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.aids.xgb.equal` | 0.0588 | [0.0559, 0.0620] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.aids.xgb.equal` | 0.0080 | [0.0030, 0.0126] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.aids.xgb.dir0.5` | -0.0603 | [-0.0636, -0.0567] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.aids.xgb.dir0.5` | -0.1303 | [-0.1358, -0.1246] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.aids.xgb.dir0.5` | -0.1176 | [-0.1220, -0.1131] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.aids.xgb.dir0.5` | -0.1259 | [-0.1313, -0.1203] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.aids.xgb.dir0.5` | -0.1375 | [-0.1430, -0.1319] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.aids.xgb.dir0.5` | -0.1068 | [-0.1122, -0.1011] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.aids.xgb.dir0.5` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.aids.xgb.dir0.5` | -0.0147 | [-0.0189, -0.0110] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.aids.xgb.dir0.5` | 0.0631 | [0.0598, 0.0664] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.aids.xgb.dir0.5` | 0.0631 | [0.0598, 0.0664] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.aids.xgb.dir0.5` | 0.0618 | [0.0585, 0.0653] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.aids.xgb.dir0.5` | 0.0278 | [0.0217, 0.0336] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.aids.xgb.dir0.1` | -0.1078 | [-0.1116, -0.1042] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.aids.xgb.dir0.1` | -0.1836 | [-0.1892, -0.1779] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.aids.xgb.dir0.1` | -0.1512 | [-0.1560, -0.1464] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.aids.xgb.dir0.1` | -0.1819 | [-0.1871, -0.1767] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.aids.xgb.dir0.1` | -0.1885 | [-0.1940, -0.1831] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.aids.xgb.dir0.1` | -0.1615 | [-0.1668, -0.1562] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.aids.xgb.dir0.1` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.aids.xgb.dir0.1` | -0.0110 | [-0.0147, -0.0074] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.aids.xgb.dir0.1` | 0.0631 | [0.0596, 0.0665] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.aids.xgb.dir0.1` | 0.0631 | [0.0596, 0.0665] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.aids.xgb.dir0.1` | 0.0520 | [0.0459, 0.0574] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.aids.xgb.dir0.1` | 0.0547 | [0.0477, 0.0617] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.diabetic.rf.equal` | -0.0000 | [-0.0000, -0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.diabetic.rf.equal` | 0.0024 | [0.0011, 0.0040] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.diabetic.rf.equal` | -0.0157 | [-0.0160, -0.0153] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.diabetic.rf.equal` | -0.0157 | [-0.0160, -0.0153] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.diabetic.rf.equal` | -0.0112 | [-0.0130, -0.0090] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.diabetic.rf.equal` | -0.0231 | [-0.0246, -0.0216] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.diabetic.rf.equal` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.diabetic.rf.equal` | -0.0045 | [-0.0092, -0.0014] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.diabetic.rf.equal` | 0.0407 | [0.0381, 0.0433] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.diabetic.rf.equal` | 0.0408 | [0.0382, 0.0434] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.diabetic.rf.equal` | 0.0395 | [0.0358, 0.0428] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.diabetic.rf.equal` | 0.0588 | [0.0546, 0.0626] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.diabetic.rf.dir0.5` | -0.0539 | [-0.0564, -0.0514] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.diabetic.rf.dir0.5` | -0.0695 | [-0.0727, -0.0660] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.diabetic.rf.dir0.5` | -0.0818 | [-0.0849, -0.0786] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.diabetic.rf.dir0.5` | -0.0866 | [-0.0897, -0.0835] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.diabetic.rf.dir0.5` | -0.0859 | [-0.0898, -0.0820] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.diabetic.rf.dir0.5` | -0.1036 | [-0.1075, -0.0998] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.diabetic.rf.dir0.5` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.diabetic.rf.dir0.5` | -0.0088 | [-0.0141, -0.0047] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.diabetic.rf.dir0.5` | 0.0631 | [0.0596, 0.0663] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.diabetic.rf.dir0.5` | 0.0552 | [0.0524, 0.0581] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.diabetic.rf.dir0.5` | 0.0620 | [0.0580, 0.0658] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.diabetic.rf.dir0.5` | 0.0998 | [0.0947, 0.1047] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.diabetic.rf.dir0.1` | -0.0593 | [-0.0622, -0.0566] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.diabetic.rf.dir0.1` | -0.1016 | [-0.1059, -0.0974] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.diabetic.rf.dir0.1` | -0.1076 | [-0.1115, -0.1036] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.diabetic.rf.dir0.1` | -0.1131 | [-0.1176, -0.1089] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.diabetic.rf.dir0.1` | -0.1115 | [-0.1160, -0.1070] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.diabetic.rf.dir0.1` | -0.1430 | [-0.1484, -0.1377] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.diabetic.rf.dir0.1` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.diabetic.rf.dir0.1` | -0.0093 | [-0.0138, -0.0056] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.diabetic.rf.dir0.1` | 0.0625 | [0.0592, 0.0659] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.diabetic.rf.dir0.1` | 0.0631 | [0.0600, 0.0661] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.diabetic.rf.dir0.1` | 0.0571 | [0.0506, 0.0627] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.diabetic.rf.dir0.1` | 0.1459 | [0.1395, 0.1518] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.diabetic.xgb.equal` | -0.0000 | [-0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.diabetic.xgb.equal` | -0.0442 | [-0.0464, -0.0420] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.diabetic.xgb.equal` | -0.0609 | [-0.0622, -0.0595] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.diabetic.xgb.equal` | -0.0609 | [-0.0622, -0.0595] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.diabetic.xgb.equal` | -0.0676 | [-0.0691, -0.0659] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.diabetic.xgb.equal` | -0.0546 | [-0.0580, -0.0507] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.diabetic.xgb.equal` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.diabetic.xgb.equal` | -0.0059 | [-0.0102, -0.0028] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.diabetic.xgb.equal` | 0.0683 | [0.0648, 0.0718] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.diabetic.xgb.equal` | 0.0683 | [0.0648, 0.0718] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.diabetic.xgb.equal` | 0.0683 | [0.0648, 0.0718] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.diabetic.xgb.equal` | 0.0859 | [0.0807, 0.0905] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.diabetic.xgb.dir0.5` | -0.0453 | [-0.0473, -0.0432] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.diabetic.xgb.dir0.5` | -0.1137 | [-0.1169, -0.1103] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.diabetic.xgb.dir0.5` | -0.1209 | [-0.1243, -0.1175] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.diabetic.xgb.dir0.5` | -0.1315 | [-0.1346, -0.1280] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.diabetic.xgb.dir0.5` | -0.1372 | [-0.1403, -0.1340] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.diabetic.xgb.dir0.5` | -0.1330 | [-0.1380, -0.1272] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.diabetic.xgb.dir0.5` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.diabetic.xgb.dir0.5` | -0.0098 | [-0.0134, -0.0066] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.diabetic.xgb.dir0.5` | 0.0881 | [0.0841, 0.0921] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.diabetic.xgb.dir0.5` | 0.0881 | [0.0841, 0.0921] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.diabetic.xgb.dir0.5` | 0.0871 | [0.0830, 0.0913] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.diabetic.xgb.dir0.5` | 0.1317 | [0.1256, 0.1371] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.diabetic.xgb.dir0.1` | -0.0593 | [-0.0622, -0.0565] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.diabetic.xgb.dir0.1` | -0.1288 | [-0.1331, -0.1246] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.diabetic.xgb.dir0.1` | -0.1274 | [-0.1312, -0.1237] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.diabetic.xgb.dir0.1` | -0.1442 | [-0.1482, -0.1405] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.diabetic.xgb.dir0.1` | -0.1471 | [-0.1513, -0.1430] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.diabetic.xgb.dir0.1` | -0.1555 | [-0.1624, -0.1483] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.diabetic.xgb.dir0.1` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.diabetic.xgb.dir0.1` | -0.0104 | [-0.0146, -0.0067] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.diabetic.xgb.dir0.1` | 0.0975 | [0.0933, 0.1015] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.diabetic.xgb.dir0.1` | 0.0975 | [0.0933, 0.1015] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.diabetic.xgb.dir0.1` | 0.0915 | [0.0850, 0.0973] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.diabetic.xgb.dir0.1` | 0.1803 | [0.1724, 0.1878] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.heart.rf.equal` | -0.0000 | [-0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.heart.rf.equal` | -0.0002 | [-0.0002, -0.0001] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.heart.rf.equal` | -0.0004 | [-0.0004, -0.0004] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.heart.rf.equal` | -0.0004 | [-0.0004, -0.0004] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.heart.rf.equal` | -0.0005 | [-0.0005, -0.0005] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.heart.rf.equal` | -0.0004 | [-0.0004, -0.0004] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.heart.rf.equal` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.heart.rf.equal` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.heart.rf.equal` | 0.0074 | [0.0073, 0.0076] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.heart.rf.equal` | 0.0074 | [0.0073, 0.0076] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.heart.rf.equal` | 0.0074 | [0.0073, 0.0076] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.heart.rf.equal` | 0.0071 | [0.0067, 0.0076] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.heart.rf.dir0.5` | -0.0553 | [-0.0611, -0.0500] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.heart.rf.dir0.5` | -0.0628 | [-0.0695, -0.0564] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.heart.rf.dir0.5` | -0.0459 | [-0.0499, -0.0419] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.heart.rf.dir0.5` | -0.0608 | [-0.0669, -0.0550] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.heart.rf.dir0.5` | -0.0642 | [-0.0708, -0.0577] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.heart.rf.dir0.5` | -0.0642 | [-0.0709, -0.0577] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.heart.rf.dir0.5` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.heart.rf.dir0.5` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.heart.rf.dir0.5` | 0.0565 | [0.0538, 0.0593] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.heart.rf.dir0.5` | 0.0413 | [0.0387, 0.0439] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.heart.rf.dir0.5` | 0.0565 | [0.0538, 0.0593] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.heart.rf.dir0.5` | 0.0520 | [0.0489, 0.0552] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.heart.rf.dir0.1` | -0.1331 | [-0.1392, -0.1270] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.heart.rf.dir0.1` | -0.2213 | [-0.2321, -0.2108] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.heart.rf.dir0.1` | -0.1261 | [-0.1312, -0.1210] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.heart.rf.dir0.1` | -0.1893 | [-0.1971, -0.1816] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.heart.rf.dir0.1` | -0.2229 | [-0.2337, -0.2124] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.heart.rf.dir0.1` | -0.2245 | [-0.2353, -0.2139] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.heart.rf.dir0.1` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.heart.rf.dir0.1` | 0.0003 | [0.0000, 0.0006] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.heart.rf.dir0.1` | 0.1041 | [0.1010, 0.1071] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.heart.rf.dir0.1` | 0.0887 | [0.0857, 0.0915] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.heart.rf.dir0.1` | 0.1041 | [0.1010, 0.1071] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.heart.rf.dir0.1` | 0.1426 | [0.1380, 0.1475] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.heart.xgb.equal` | 0.0000 | [-0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.heart.xgb.equal` | -0.0017 | [-0.0017, -0.0016] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.heart.xgb.equal` | -0.0027 | [-0.0027, -0.0027] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.heart.xgb.equal` | -0.0027 | [-0.0027, -0.0027] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.heart.xgb.equal` | -0.0032 | [-0.0032, -0.0032] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.heart.xgb.equal` | -0.0026 | [-0.0026, -0.0025] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.heart.xgb.equal` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.heart.xgb.equal` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.heart.xgb.equal` | 0.0293 | [0.0291, 0.0294] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.heart.xgb.equal` | 0.0293 | [0.0291, 0.0294] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.heart.xgb.equal` | 0.0293 | [0.0291, 0.0294] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.heart.xgb.equal` | 0.0274 | [0.0269, 0.0278] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.heart.xgb.dir0.5` | -0.0576 | [-0.0632, -0.0522] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.heart.xgb.dir0.5` | -0.0713 | [-0.0785, -0.0645] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.heart.xgb.dir0.5` | -0.0641 | [-0.0695, -0.0586] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.heart.xgb.dir0.5` | -0.0711 | [-0.0780, -0.0644] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.heart.xgb.dir0.5` | -0.0731 | [-0.0803, -0.0663] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.heart.xgb.dir0.5` | -0.0730 | [-0.0802, -0.0662] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.heart.xgb.dir0.5` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.heart.xgb.dir0.5` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.heart.xgb.dir0.5` | 0.0485 | [0.0473, 0.0497] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.heart.xgb.dir0.5` | 0.0485 | [0.0473, 0.0497] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.heart.xgb.dir0.5` | 0.0485 | [0.0473, 0.0497] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.heart.xgb.dir0.5` | 0.0527 | [0.0508, 0.0546] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L1.heart.xgb.dir0.1` | -0.1935 | [-0.2021, -0.1850] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L2.heart.xgb.dir0.1` | -0.2353 | [-0.2463, -0.2248] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3.heart.xgb.dir0.1` | -0.1808 | [-0.1916, -0.1696] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L1.heart.xgb.dir0.1` | -0.2342 | [-0.2449, -0.2238] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.L3+L2.heart.xgb.dir0.1` | -0.2372 | [-0.2482, -0.2267] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.brier.Central.heart.xgb.dir0.1` | -0.2384 | [-0.2494, -0.2279] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L1.heart.xgb.dir0.1` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L2.heart.xgb.dir0.1` | 0.0007 | [0.0003, 0.0011] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3.heart.xgb.dir0.1` | 0.0920 | [0.0904, 0.0937] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L1.heart.xgb.dir0.1` | 0.0920 | [0.0904, 0.0937] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.L3+L2.heart.xgb.dir0.1` | 0.0920 | [0.0904, 0.0937] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `diff.auroc.Central.heart.xgb.dir0.1` | 0.1220 | [0.1194, 0.1248] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level minus L0 (mean source model). |
| `level.brier.L0.aids.rf.equal` | 0.1787 | [0.1774, 0.1800] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.aids.rf.equal` | 0.1787 | [0.1774, 0.1800] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.aids.rf.equal` | 0.1776 | [0.1766, 0.1786] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.aids.rf.equal` | 0.1702 | [0.1689, 0.1716] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.aids.rf.equal` | 0.1702 | [0.1689, 0.1716] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.aids.rf.equal` | 0.1725 | [0.1710, 0.1741] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.aids.rf.equal` | 0.1745 | [0.1726, 0.1763] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.aids.rf.equal` | 0.6381 | [0.6319, 0.6443] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.aids.rf.equal` | 0.6381 | [0.6319, 0.6443] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.aids.rf.equal` | 0.6280 | [0.6210, 0.6350] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.aids.rf.equal` | 0.6721 | [0.6640, 0.6799] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.aids.rf.equal` | 0.6721 | [0.6641, 0.6799] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.aids.rf.equal` | 0.6711 | [0.6630, 0.6790] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.aids.rf.equal` | 0.6589 | [0.6502, 0.6679] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.aids.rf.dir0.5` | 0.2680 | [0.2626, 0.2732] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.aids.rf.dir0.5` | 0.1948 | [0.1931, 0.1969] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.aids.rf.dir0.5` | 0.1803 | [0.1796, 0.1811] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.aids.rf.dir0.5` | 0.1900 | [0.1876, 0.1922] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.aids.rf.dir0.5` | 0.1767 | [0.1754, 0.1780] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.aids.rf.dir0.5` | 0.1748 | [0.1736, 0.1761] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.aids.rf.dir0.5` | 0.1745 | [0.1726, 0.1763] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.aids.rf.dir0.5` | 0.6085 | [0.6035, 0.6132] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.aids.rf.dir0.5` | 0.6085 | [0.6035, 0.6132] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.aids.rf.dir0.5` | 0.5971 | [0.5915, 0.6024] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.aids.rf.dir0.5` | 0.6543 | [0.6472, 0.6612] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.aids.rf.dir0.5` | 0.6454 | [0.6386, 0.6521] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.aids.rf.dir0.5` | 0.6511 | [0.6441, 0.6583] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.aids.rf.dir0.5` | 0.6589 | [0.6502, 0.6679] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.aids.rf.dir0.1` | 0.3364 | [0.3305, 0.3420] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.aids.rf.dir0.1` | 0.2288 | [0.2260, 0.2318] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.aids.rf.dir0.1` | 0.1827 | [0.1823, 0.1832] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.aids.rf.dir0.1` | 0.2236 | [0.2202, 0.2272] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.aids.rf.dir0.1` | 0.1895 | [0.1879, 0.1911] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.aids.rf.dir0.1` | 0.1790 | [0.1783, 0.1798] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.aids.rf.dir0.1` | 0.1745 | [0.1726, 0.1763] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.aids.rf.dir0.1` | 0.5724 | [0.5686, 0.5762] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.aids.rf.dir0.1` | 0.5724 | [0.5686, 0.5762] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.aids.rf.dir0.1` | 0.5606 | [0.5565, 0.5646] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.aids.rf.dir0.1` | 0.6198 | [0.6135, 0.6262] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.aids.rf.dir0.1` | 0.6106 | [0.6045, 0.6170] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.aids.rf.dir0.1` | 0.6038 | [0.5965, 0.6110] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.aids.rf.dir0.1` | 0.6589 | [0.6502, 0.6679] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.aids.xgb.equal` | 0.2207 | [0.2186, 0.2227] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.aids.xgb.equal` | 0.2207 | [0.2186, 0.2227] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.aids.xgb.equal` | 0.1804 | [0.1797, 0.1810] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.aids.xgb.equal` | 0.1883 | [0.1859, 0.1905] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.aids.xgb.equal` | 0.1883 | [0.1859, 0.1905] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.aids.xgb.equal` | 0.1722 | [0.1706, 0.1737] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.aids.xgb.equal` | 0.2055 | [0.2025, 0.2086] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.aids.xgb.equal` | 0.6101 | [0.6046, 0.6158] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.aids.xgb.equal` | 0.6101 | [0.6046, 0.6158] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.aids.xgb.equal` | 0.5978 | [0.5920, 0.6038] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.aids.xgb.equal` | 0.6689 | [0.6606, 0.6776] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.aids.xgb.equal` | 0.6689 | [0.6606, 0.6776] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.aids.xgb.equal` | 0.6689 | [0.6606, 0.6776] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.aids.xgb.equal` | 0.6181 | [0.6095, 0.6269] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.aids.xgb.dir0.5` | 0.3123 | [0.3064, 0.3179] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.aids.xgb.dir0.5` | 0.2520 | [0.2492, 0.2548] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.aids.xgb.dir0.5` | 0.1820 | [0.1814, 0.1826] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.aids.xgb.dir0.5` | 0.1947 | [0.1921, 0.1975] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.aids.xgb.dir0.5` | 0.1864 | [0.1846, 0.1883] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.aids.xgb.dir0.5` | 0.1749 | [0.1737, 0.1761] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.aids.xgb.dir0.5` | 0.2055 | [0.2025, 0.2086] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.aids.xgb.dir0.5` | 0.5903 | [0.5856, 0.5947] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.aids.xgb.dir0.5` | 0.5903 | [0.5856, 0.5947] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.aids.xgb.dir0.5` | 0.5756 | [0.5708, 0.5801] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.aids.xgb.dir0.5` | 0.6534 | [0.6458, 0.6609] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.aids.xgb.dir0.5` | 0.6534 | [0.6458, 0.6609] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.aids.xgb.dir0.5` | 0.6522 | [0.6445, 0.6596] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.aids.xgb.dir0.5` | 0.6181 | [0.6095, 0.6269] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.aids.xgb.dir0.1` | 0.3670 | [0.3613, 0.3728] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.aids.xgb.dir0.1` | 0.2592 | [0.2566, 0.2619] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.aids.xgb.dir0.1` | 0.1835 | [0.1831, 0.1839] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.aids.xgb.dir0.1` | 0.2159 | [0.2111, 0.2210] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.aids.xgb.dir0.1` | 0.1851 | [0.1836, 0.1866] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.aids.xgb.dir0.1` | 0.1785 | [0.1777, 0.1793] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.aids.xgb.dir0.1` | 0.2055 | [0.2025, 0.2086] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.aids.xgb.dir0.1` | 0.5634 | [0.5599, 0.5672] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.aids.xgb.dir0.1` | 0.5634 | [0.5599, 0.5672] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.aids.xgb.dir0.1` | 0.5524 | [0.5483, 0.5565] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.aids.xgb.dir0.1` | 0.6264 | [0.6198, 0.6330] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.aids.xgb.dir0.1` | 0.6264 | [0.6198, 0.6330] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.aids.xgb.dir0.1` | 0.6154 | [0.6074, 0.6229] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.aids.xgb.dir0.1` | 0.6181 | [0.6095, 0.6269] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.diabetic.rf.equal` | 0.2236 | [0.2214, 0.2260] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.diabetic.rf.equal` | 0.2236 | [0.2214, 0.2260] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.diabetic.rf.equal` | 0.2260 | [0.2239, 0.2284] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.diabetic.rf.equal` | 0.2080 | [0.2056, 0.2103] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.diabetic.rf.equal` | 0.2080 | [0.2056, 0.2103] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.diabetic.rf.equal` | 0.2124 | [0.2089, 0.2161] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.diabetic.rf.equal` | 0.2005 | [0.1976, 0.2036] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.diabetic.rf.equal` | 0.6911 | [0.6846, 0.6974] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.diabetic.rf.equal` | 0.6911 | [0.6846, 0.6974] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.diabetic.rf.equal` | 0.6866 | [0.6796, 0.6931] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.diabetic.rf.equal` | 0.7317 | [0.7233, 0.7401] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.diabetic.rf.equal` | 0.7318 | [0.7234, 0.7402] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.diabetic.rf.equal` | 0.7306 | [0.7219, 0.7390] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.diabetic.rf.equal` | 0.7498 | [0.7414, 0.7575] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.diabetic.rf.dir0.5` | 0.3041 | [0.3002, 0.3079] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.diabetic.rf.dir0.5` | 0.2502 | [0.2477, 0.2527] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.diabetic.rf.dir0.5` | 0.2346 | [0.2330, 0.2364] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.diabetic.rf.dir0.5` | 0.2223 | [0.2205, 0.2243] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.diabetic.rf.dir0.5` | 0.2175 | [0.2154, 0.2197] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.diabetic.rf.dir0.5` | 0.2182 | [0.2154, 0.2212] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.diabetic.rf.dir0.5` | 0.2005 | [0.1976, 0.2036] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.diabetic.rf.dir0.5` | 0.6500 | [0.6446, 0.6552] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.diabetic.rf.dir0.5` | 0.6500 | [0.6446, 0.6552] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.diabetic.rf.dir0.5` | 0.6412 | [0.6344, 0.6476] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.diabetic.rf.dir0.5` | 0.7131 | [0.7051, 0.7206] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.diabetic.rf.dir0.5` | 0.7052 | [0.6977, 0.7126] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.diabetic.rf.dir0.5` | 0.7120 | [0.7041, 0.7199] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.diabetic.rf.dir0.5` | 0.7498 | [0.7414, 0.7575] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.diabetic.rf.dir0.1` | 0.3435 | [0.3390, 0.3481] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.diabetic.rf.dir0.1` | 0.2842 | [0.2813, 0.2871] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.diabetic.rf.dir0.1` | 0.2420 | [0.2408, 0.2432] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.diabetic.rf.dir0.1` | 0.2359 | [0.2346, 0.2374] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.diabetic.rf.dir0.1` | 0.2304 | [0.2286, 0.2322] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.diabetic.rf.dir0.1` | 0.2320 | [0.2298, 0.2346] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.diabetic.rf.dir0.1` | 0.2005 | [0.1976, 0.2036] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.diabetic.rf.dir0.1` | 0.6040 | [0.5995, 0.6085] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.diabetic.rf.dir0.1` | 0.6040 | [0.5995, 0.6085] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.diabetic.rf.dir0.1` | 0.5946 | [0.5896, 0.5998] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.diabetic.rf.dir0.1` | 0.6665 | [0.6596, 0.6732] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.diabetic.rf.dir0.1` | 0.6670 | [0.6604, 0.6735] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.diabetic.rf.dir0.1` | 0.6610 | [0.6523, 0.6691] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.diabetic.rf.dir0.1` | 0.7498 | [0.7414, 0.7575] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.diabetic.xgb.equal` | 0.2714 | [0.2679, 0.2749] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.diabetic.xgb.equal` | 0.2714 | [0.2679, 0.2749] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.diabetic.xgb.equal` | 0.2272 | [0.2256, 0.2289] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.diabetic.xgb.equal` | 0.2105 | [0.2060, 0.2150] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.diabetic.xgb.equal` | 0.2105 | [0.2060, 0.2150] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.diabetic.xgb.equal` | 0.2038 | [0.2003, 0.2074] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.diabetic.xgb.equal` | 0.2168 | [0.2114, 0.2226] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.diabetic.xgb.equal` | 0.6864 | [0.6808, 0.6918] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.diabetic.xgb.equal` | 0.6864 | [0.6808, 0.6918] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.diabetic.xgb.equal` | 0.6805 | [0.6744, 0.6862] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.diabetic.xgb.equal` | 0.7547 | [0.7461, 0.7631] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.diabetic.xgb.equal` | 0.7547 | [0.7461, 0.7631] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.diabetic.xgb.equal` | 0.7547 | [0.7461, 0.7631] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.diabetic.xgb.equal` | 0.7723 | [0.7638, 0.7804] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.diabetic.xgb.dir0.5` | 0.3498 | [0.3457, 0.3536] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.diabetic.xgb.dir0.5` | 0.3045 | [0.3016, 0.3075] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.diabetic.xgb.dir0.5` | 0.2361 | [0.2347, 0.2374] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.diabetic.xgb.dir0.5` | 0.2289 | [0.2249, 0.2333] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.diabetic.xgb.dir0.5` | 0.2183 | [0.2146, 0.2221] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.diabetic.xgb.dir0.5` | 0.2126 | [0.2096, 0.2157] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.diabetic.xgb.dir0.5` | 0.2168 | [0.2114, 0.2226] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.diabetic.xgb.dir0.5` | 0.6406 | [0.6353, 0.6456] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.diabetic.xgb.dir0.5` | 0.6406 | [0.6353, 0.6456] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.diabetic.xgb.dir0.5` | 0.6307 | [0.6254, 0.6361] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.diabetic.xgb.dir0.5` | 0.7287 | [0.7204, 0.7370] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.diabetic.xgb.dir0.5` | 0.7287 | [0.7204, 0.7370] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.diabetic.xgb.dir0.5` | 0.7277 | [0.7192, 0.7357] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.diabetic.xgb.dir0.5` | 0.7723 | [0.7638, 0.7804] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.diabetic.xgb.dir0.1` | 0.3723 | [0.3678, 0.3770] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.diabetic.xgb.dir0.1` | 0.3130 | [0.3105, 0.3155] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.diabetic.xgb.dir0.1` | 0.2435 | [0.2425, 0.2445] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.diabetic.xgb.dir0.1` | 0.2449 | [0.2420, 0.2481] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.diabetic.xgb.dir0.1` | 0.2281 | [0.2255, 0.2307] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.diabetic.xgb.dir0.1` | 0.2252 | [0.2229, 0.2276] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.diabetic.xgb.dir0.1` | 0.2168 | [0.2114, 0.2226] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.diabetic.xgb.dir0.1` | 0.5920 | [0.5883, 0.5958] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.diabetic.xgb.dir0.1` | 0.5920 | [0.5883, 0.5958] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.diabetic.xgb.dir0.1` | 0.5816 | [0.5767, 0.5866] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.diabetic.xgb.dir0.1` | 0.6894 | [0.6822, 0.6966] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.diabetic.xgb.dir0.1` | 0.6894 | [0.6822, 0.6966] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.diabetic.xgb.dir0.1` | 0.6835 | [0.6748, 0.6922] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.diabetic.xgb.dir0.1` | 0.7723 | [0.7638, 0.7804] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.heart.rf.equal` | 0.0468 | [0.0467, 0.0468] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.heart.rf.equal` | 0.0468 | [0.0467, 0.0468] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.heart.rf.equal` | 0.0466 | [0.0466, 0.0467] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.heart.rf.equal` | 0.0464 | [0.0463, 0.0464] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.heart.rf.equal` | 0.0464 | [0.0463, 0.0464] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.heart.rf.equal` | 0.0463 | [0.0462, 0.0463] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.heart.rf.equal` | 0.0464 | [0.0463, 0.0464] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.heart.rf.equal` | 0.8181 | [0.8170, 0.8193] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.heart.rf.equal` | 0.8181 | [0.8170, 0.8193] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.heart.rf.equal` | 0.8181 | [0.8170, 0.8193] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.heart.rf.equal` | 0.8256 | [0.8244, 0.8267] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.heart.rf.equal` | 0.8256 | [0.8244, 0.8267] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.heart.rf.equal` | 0.8256 | [0.8244, 0.8267] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.heart.rf.equal` | 0.8252 | [0.8240, 0.8265] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.heart.rf.dir0.5` | 0.1105 | [0.1041, 0.1173] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.heart.rf.dir0.5` | 0.0552 | [0.0534, 0.0573] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.heart.rf.dir0.5` | 0.0477 | [0.0476, 0.0478] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.heart.rf.dir0.5` | 0.0646 | [0.0620, 0.0674] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.heart.rf.dir0.5` | 0.0497 | [0.0490, 0.0505] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.heart.rf.dir0.5` | 0.0464 | [0.0463, 0.0464] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.heart.rf.dir0.5` | 0.0464 | [0.0463, 0.0464] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.heart.rf.dir0.5` | 0.7732 | [0.7701, 0.7764] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.heart.rf.dir0.5` | 0.7732 | [0.7701, 0.7764] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.heart.rf.dir0.5` | 0.7732 | [0.7701, 0.7764] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.heart.rf.dir0.5` | 0.8297 | [0.8285, 0.8310] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.heart.rf.dir0.5` | 0.8145 | [0.8129, 0.8160] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.heart.rf.dir0.5` | 0.8297 | [0.8285, 0.8310] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.heart.rf.dir0.5` | 0.8252 | [0.8240, 0.8265] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.heart.rf.dir0.1` | 0.2708 | [0.2603, 0.2817] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.heart.rf.dir0.1` | 0.1377 | [0.1298, 0.1458] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.heart.rf.dir0.1` | 0.0496 | [0.0495, 0.0497] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.heart.rf.dir0.1` | 0.1448 | [0.1380, 0.1522] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.heart.rf.dir0.1` | 0.0816 | [0.0781, 0.0853] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.heart.rf.dir0.1` | 0.0479 | [0.0478, 0.0480] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.heart.rf.dir0.1` | 0.0464 | [0.0463, 0.0464] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.heart.rf.dir0.1` | 0.6826 | [0.6779, 0.6871] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.heart.rf.dir0.1` | 0.6826 | [0.6779, 0.6871] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.heart.rf.dir0.1` | 0.6829 | [0.6782, 0.6874] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.heart.rf.dir0.1` | 0.7868 | [0.7822, 0.7908] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.heart.rf.dir0.1` | 0.7713 | [0.7670, 0.7750] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.heart.rf.dir0.1` | 0.7868 | [0.7822, 0.7908] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.heart.rf.dir0.1` | 0.8252 | [0.8240, 0.8265] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.heart.xgb.equal` | 0.0491 | [0.0490, 0.0492] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.heart.xgb.equal` | 0.0491 | [0.0490, 0.0492] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.heart.xgb.equal` | 0.0474 | [0.0474, 0.0475] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.heart.xgb.equal` | 0.0464 | [0.0463, 0.0464] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.heart.xgb.equal` | 0.0464 | [0.0463, 0.0464] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.heart.xgb.equal` | 0.0459 | [0.0458, 0.0459] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.heart.xgb.equal` | 0.0465 | [0.0464, 0.0466] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.heart.xgb.equal` | 0.8092 | [0.8081, 0.8102] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.heart.xgb.equal` | 0.8092 | [0.8081, 0.8102] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.heart.xgb.equal` | 0.8092 | [0.8081, 0.8102] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.heart.xgb.equal` | 0.8384 | [0.8373, 0.8395] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.heart.xgb.equal` | 0.8384 | [0.8373, 0.8395] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.heart.xgb.equal` | 0.8384 | [0.8373, 0.8395] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.heart.xgb.equal` | 0.8365 | [0.8354, 0.8377] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.heart.xgb.dir0.5` | 0.1195 | [0.1127, 0.1267] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.heart.xgb.dir0.5` | 0.0619 | [0.0604, 0.0637] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.heart.xgb.dir0.5` | 0.0482 | [0.0481, 0.0482] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.heart.xgb.dir0.5` | 0.0553 | [0.0529, 0.0581] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.heart.xgb.dir0.5` | 0.0484 | [0.0481, 0.0487] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.heart.xgb.dir0.5` | 0.0464 | [0.0463, 0.0464] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.heart.xgb.dir0.5` | 0.0465 | [0.0464, 0.0466] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.heart.xgb.dir0.5` | 0.7838 | [0.7820, 0.7858] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.heart.xgb.dir0.5` | 0.7838 | [0.7820, 0.7858] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.heart.xgb.dir0.5` | 0.7838 | [0.7820, 0.7858] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.heart.xgb.dir0.5` | 0.8323 | [0.8312, 0.8335] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.heart.xgb.dir0.5` | 0.8323 | [0.8312, 0.8335] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.heart.xgb.dir0.5` | 0.8323 | [0.8312, 0.8335] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.heart.xgb.dir0.5` | 0.8365 | [0.8354, 0.8377] | 50 | Mean over draws, K averaged. |
| `level.brier.L0.heart.xgb.dir0.1` | 0.2849 | [0.2743, 0.2959] | 50 | Mean over draws, K averaged. |
| `level.brier.L1.heart.xgb.dir0.1` | 0.0914 | [0.0885, 0.0943] | 50 | Mean over draws, K averaged. |
| `level.brier.L2.heart.xgb.dir0.1` | 0.0496 | [0.0495, 0.0496] | 50 | Mean over draws, K averaged. |
| `level.brier.L3.heart.xgb.dir0.1` | 0.1041 | [0.0928, 0.1170] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L1.heart.xgb.dir0.1` | 0.0506 | [0.0502, 0.0512] | 50 | Mean over draws, K averaged. |
| `level.brier.L3+L2.heart.xgb.dir0.1` | 0.0476 | [0.0476, 0.0477] | 50 | Mean over draws, K averaged. |
| `level.brier.Central.heart.xgb.dir0.1` | 0.0465 | [0.0464, 0.0466] | 50 | Mean over draws, K averaged. |
| `level.auroc.L0.heart.xgb.dir0.1` | 0.7145 | [0.7118, 0.7172] | 50 | Mean over draws, K averaged. |
| `level.auroc.L1.heart.xgb.dir0.1` | 0.7145 | [0.7118, 0.7172] | 50 | Mean over draws, K averaged. |
| `level.auroc.L2.heart.xgb.dir0.1` | 0.7152 | [0.7125, 0.7178] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3.heart.xgb.dir0.1` | 0.8065 | [0.8045, 0.8085] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L1.heart.xgb.dir0.1` | 0.8065 | [0.8045, 0.8085] | 50 | Mean over draws, K averaged. |
| `level.auroc.L3+L2.heart.xgb.dir0.1` | 0.8065 | [0.8045, 0.8085] | 50 | Mean over draws, K averaged. |
| `level.auroc.Central.heart.xgb.dir0.1` | 0.8365 | [0.8354, 0.8377] | 50 | Mean over draws, K averaged. |
| `diff.auroc.L3.equal.range` | [0.0074, 0.0683] |  |  | AUROC gain of fed_L0 over src_L0, minimum and maximum over pairs. |
| `diff.auroc.L3.dir0.5.range` | [0.0458, 0.0881] |  |  | AUROC gain of fed_L0 over src_L0, minimum and maximum over pairs. |
| `diff.auroc.L3.dir0.1.range` | [0.0474, 0.1041] |  |  | AUROC gain of fed_L0 over src_L0, minimum and maximum over pairs. |
| `perK.auroc.L1.aids.rf.equal.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.equal.K2` | 0.0171 | [0.0148, 0.0193] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.equal.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.equal.K3` | 0.0202 | [0.0173, 0.0234] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.equal.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.equal.K4` | 0.0299 | [0.0267, 0.0333] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.equal.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.equal.K5` | 0.0327 | [0.0285, 0.0370] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.equal.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.equal.K6` | 0.0365 | [0.0325, 0.0404] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.equal.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.equal.K7` | 0.0409 | [0.0369, 0.0450] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.equal.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.equal.K8` | 0.0394 | [0.0360, 0.0428] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.equal.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.equal.K9` | 0.0426 | [0.0385, 0.0468] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.equal.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.equal.K10` | 0.0464 | [0.0415, 0.0513] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.equal.K2` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.equal.K2` | -0.0046 | [-0.0049, -0.0042] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.equal.K3` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.equal.K3` | -0.0059 | [-0.0064, -0.0054] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.equal.K4` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.equal.K4` | -0.0075 | [-0.0079, -0.0070] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.equal.K5` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.equal.K5` | -0.0082 | [-0.0088, -0.0077] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.equal.K6` | -0.0000 | [-0.0000, -0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.equal.K6` | -0.0092 | [-0.0098, -0.0086] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.equal.K7` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.equal.K7` | -0.0098 | [-0.0104, -0.0093] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.equal.K8` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.equal.K8` | -0.0096 | [-0.0100, -0.0092] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.equal.K9` | -0.0000 | [-0.0000, -0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.equal.K9` | -0.0102 | [-0.0107, -0.0097] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.equal.K10` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.equal.K10` | -0.0109 | [-0.0115, -0.0103] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.5.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.5.K2` | 0.0228 | [0.0175, 0.0284] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.5.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.5.K3` | 0.0387 | [0.0330, 0.0448] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.5.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.5.K4` | 0.0449 | [0.0392, 0.0511] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.5.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.5.K5` | 0.0418 | [0.0364, 0.0476] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.5.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.5.K6` | 0.0507 | [0.0441, 0.0571] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.5.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.5.K7` | 0.0506 | [0.0441, 0.0574] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.5.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.5.K8` | 0.0568 | [0.0519, 0.0615] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.5.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.5.K9` | 0.0535 | [0.0469, 0.0602] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.5.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.5.K10` | 0.0524 | [0.0457, 0.0590] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.5.K2` | -0.0556 | [-0.0737, -0.0369] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.5.K2` | -0.0511 | [-0.0657, -0.0370] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.5.K3` | -0.0851 | [-0.1028, -0.0685] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.5.K3` | -0.0902 | [-0.1067, -0.0747] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.5.K4` | -0.0788 | [-0.0923, -0.0652] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.5.K4` | -0.0858 | [-0.0986, -0.0730] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.5.K5` | -0.0626 | [-0.0740, -0.0523] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.5.K5` | -0.0700 | [-0.0793, -0.0612] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.5.K6` | -0.0879 | [-0.0996, -0.0766] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.5.K6` | -0.0892 | [-0.0985, -0.0797] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.5.K7` | -0.0854 | [-0.0952, -0.0760] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.5.K7` | -0.0834 | [-0.0907, -0.0761] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.5.K8` | -0.0702 | [-0.0784, -0.0622] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.5.K8` | -0.0806 | [-0.0885, -0.0726] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.5.K9` | -0.0624 | [-0.0702, -0.0550] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.5.K9` | -0.0733 | [-0.0805, -0.0666] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.5.K10` | -0.0703 | [-0.0774, -0.0634] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.5.K10` | -0.0785 | [-0.0852, -0.0723] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.1.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.1.K2` | 0.0228 | [0.0175, 0.0285] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.1.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.1.K3` | 0.0405 | [0.0327, 0.0487] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.1.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.1.K4` | 0.0371 | [0.0275, 0.0461] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.1.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.1.K5` | 0.0499 | [0.0403, 0.0592] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.1.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.1.K6` | 0.0491 | [0.0399, 0.0575] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.1.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.1.K7` | 0.0522 | [0.0440, 0.0600] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.1.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.1.K8` | 0.0544 | [0.0467, 0.0618] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.1.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.1.K9` | 0.0531 | [0.0440, 0.0615] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.rf.dir0.1.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.rf.dir0.1.K10` | 0.0676 | [0.0593, 0.0762] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.1.K2` | -0.1073 | [-0.1255, -0.0898] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.1.K2` | -0.1353 | [-0.1613, -0.1095] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.1.K3` | -0.0994 | [-0.1147, -0.0836] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.1.K3` | -0.1189 | [-0.1360, -0.1012] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.1.K4` | -0.1190 | [-0.1338, -0.1037] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.1.K4` | -0.1274 | [-0.1425, -0.1124] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.1.K5` | -0.0993 | [-0.1123, -0.0876] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.1.K5` | -0.1075 | [-0.1190, -0.0961] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.1.K6` | -0.1103 | [-0.1216, -0.1004] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.1.K6` | -0.1155 | [-0.1251, -0.1059] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.1.K7` | -0.1071 | [-0.1195, -0.0943] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.1.K7` | -0.1032 | [-0.1124, -0.0937] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.1.K8` | -0.1084 | [-0.1179, -0.0994] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.1.K8` | -0.1028 | [-0.1097, -0.0955] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.1.K9` | -0.1097 | [-0.1168, -0.1025] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.1.K9` | -0.1047 | [-0.1110, -0.0986] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.rf.dir0.1.K10` | -0.1071 | [-0.1166, -0.0980] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.rf.dir0.1.K10` | -0.1000 | [-0.1056, -0.0945] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.equal.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.equal.K2` | 0.0258 | [0.0233, 0.0281] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.equal.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.equal.K3` | 0.0422 | [0.0388, 0.0454] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.equal.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.equal.K4` | 0.0525 | [0.0492, 0.0562] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.equal.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.equal.K5` | 0.0567 | [0.0530, 0.0603] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.equal.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.equal.K6` | 0.0646 | [0.0610, 0.0685] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.equal.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.equal.K7` | 0.0670 | [0.0636, 0.0704] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.equal.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.equal.K8` | 0.0704 | [0.0661, 0.0748] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.equal.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.equal.K9` | 0.0736 | [0.0695, 0.0778] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.equal.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.equal.K10` | 0.0767 | [0.0722, 0.0812] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.equal.K2` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.equal.K2` | -0.0184 | [-0.0193, -0.0174] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.equal.K3` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.equal.K3` | -0.0265 | [-0.0275, -0.0253] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.equal.K4` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.equal.K4` | -0.0282 | [-0.0295, -0.0271] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.equal.K5` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.equal.K5` | -0.0326 | [-0.0338, -0.0314] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.equal.K6` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.equal.K6` | -0.0345 | [-0.0356, -0.0334] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.equal.K7` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.equal.K7` | -0.0353 | [-0.0364, -0.0342] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.equal.K8` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.equal.K8` | -0.0371 | [-0.0382, -0.0359] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.equal.K9` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.equal.K9` | -0.0387 | [-0.0402, -0.0373] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.equal.K10` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.equal.K10` | -0.0401 | [-0.0415, -0.0387] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.5.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.5.K2` | 0.0263 | [0.0231, 0.0296] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.5.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.5.K3` | 0.0434 | [0.0398, 0.0472] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.5.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.5.K4` | 0.0527 | [0.0480, 0.0575] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.5.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.5.K5` | 0.0574 | [0.0524, 0.0623] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.5.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.5.K6` | 0.0742 | [0.0697, 0.0783] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.5.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.5.K7` | 0.0693 | [0.0641, 0.0743] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.5.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.5.K8` | 0.0774 | [0.0725, 0.0822] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.5.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.5.K9` | 0.0816 | [0.0772, 0.0862] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.5.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.5.K10` | 0.0852 | [0.0803, 0.0906] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.5.K2` | -0.0435 | [-0.0565, -0.0300] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.5.K2` | -0.0641 | [-0.0807, -0.0479] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.5.K3` | -0.0691 | [-0.0843, -0.0556] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.5.K3` | -0.1180 | [-0.1360, -0.1004] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.5.K4` | -0.0651 | [-0.0764, -0.0539] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.5.K4` | -0.1187 | [-0.1324, -0.1046] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.5.K5` | -0.0506 | [-0.0589, -0.0431] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.5.K5` | -0.1048 | [-0.1183, -0.0924] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.5.K6` | -0.0720 | [-0.0808, -0.0625] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.5.K6` | -0.1430 | [-0.1556, -0.1295] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.5.K7` | -0.0707 | [-0.0787, -0.0632] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.5.K7` | -0.1358 | [-0.1448, -0.1265] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.5.K8` | -0.0587 | [-0.0657, -0.0519] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.5.K8` | -0.1263 | [-0.1361, -0.1160] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.5.K9` | -0.0529 | [-0.0594, -0.0466] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.5.K9` | -0.1180 | [-0.1278, -0.1080] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.5.K10` | -0.0601 | [-0.0665, -0.0539] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.5.K10` | -0.1297 | [-0.1394, -0.1202] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.1.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.1.K2` | 0.0297 | [0.0263, 0.0336] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.1.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.1.K3` | 0.0441 | [0.0389, 0.0493] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.1.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.1.K4` | 0.0555 | [0.0496, 0.0611] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.1.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.1.K5` | 0.0602 | [0.0536, 0.0663] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.1.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.1.K6` | 0.0708 | [0.0641, 0.0778] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.1.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.1.K7` | 0.0741 | [0.0671, 0.0808] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.1.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.1.K8` | 0.0772 | [0.0696, 0.0844] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.1.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.1.K9` | 0.0767 | [0.0687, 0.0843] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.aids.xgb.dir0.1.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.aids.xgb.dir0.1.K10` | 0.0792 | [0.0734, 0.0851] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.1.K2` | -0.1189 | [-0.1389, -0.0996] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.1.K2` | -0.1578 | [-0.1853, -0.1299] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.1.K3` | -0.1037 | [-0.1177, -0.0889] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.1.K3` | -0.1434 | [-0.1626, -0.1230] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.1.K4` | -0.1208 | [-0.1348, -0.1067] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.1.K4` | -0.1557 | [-0.1745, -0.1365] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.1.K5` | -0.0996 | [-0.1115, -0.0884] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.1.K5` | -0.1449 | [-0.1576, -0.1321] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.1.K6` | -0.1053 | [-0.1151, -0.0960] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.1.K6` | -0.1594 | [-0.1707, -0.1477] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.1.K7` | -0.1042 | [-0.1157, -0.0926] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.1.K7` | -0.1431 | [-0.1557, -0.1298] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.1.K8` | -0.1075 | [-0.1167, -0.0985] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.1.K8` | -0.1511 | [-0.1605, -0.1411] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.1.K9` | -0.1066 | [-0.1132, -0.0998] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.1.K9` | -0.1529 | [-0.1611, -0.1443] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.aids.xgb.dir0.1.K10` | -0.1038 | [-0.1121, -0.0957] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.aids.xgb.dir0.1.K10` | -0.1523 | [-0.1605, -0.1450] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.equal.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.equal.K2` | 0.0194 | [0.0165, 0.0225] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.equal.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.equal.K3` | 0.0269 | [0.0237, 0.0304] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.equal.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.equal.K4` | 0.0342 | [0.0300, 0.0383] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.equal.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.equal.K5` | 0.0406 | [0.0365, 0.0447] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.equal.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.equal.K6` | 0.0433 | [0.0391, 0.0473] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.equal.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.equal.K7` | 0.0494 | [0.0457, 0.0534] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.equal.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.equal.K8` | 0.0486 | [0.0442, 0.0535] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.equal.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.equal.K9` | 0.0523 | [0.0475, 0.0571] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.equal.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.equal.K10` | 0.0512 | [0.0458, 0.0566] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.equal.K2` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.equal.K2` | -0.0077 | [-0.0085, -0.0069] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.equal.K3` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.equal.K3` | -0.0102 | [-0.0110, -0.0093] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.equal.K4` | -0.0000 | [-0.0000, -0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.equal.K4` | -0.0133 | [-0.0143, -0.0123] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.equal.K5` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.equal.K5` | -0.0150 | [-0.0160, -0.0141] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.equal.K6` | -0.0000 | [-0.0000, -0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.equal.K6` | -0.0166 | [-0.0176, -0.0157] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.equal.K7` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.equal.K7` | -0.0187 | [-0.0197, -0.0177] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.equal.K8` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.equal.K8` | -0.0191 | [-0.0201, -0.0181] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.equal.K9` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.equal.K9` | -0.0203 | [-0.0212, -0.0193] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.equal.K10` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.equal.K10` | -0.0202 | [-0.0214, -0.0190] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.5.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.5.K2` | 0.0425 | [0.0355, 0.0498] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.5.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.5.K3` | 0.0459 | [0.0401, 0.0518] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.5.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.5.K4` | 0.0579 | [0.0507, 0.0653] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.5.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.5.K5` | 0.0638 | [0.0583, 0.0697] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.5.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.5.K6` | 0.0660 | [0.0599, 0.0718] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.5.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.5.K7` | 0.0684 | [0.0616, 0.0757] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.5.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.5.K8` | 0.0735 | [0.0659, 0.0810] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.5.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.5.K9` | 0.0710 | [0.0652, 0.0769] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.5.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.5.K10` | 0.0785 | [0.0694, 0.0875] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.5.K2` | -0.0639 | [-0.0763, -0.0512] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.5.K2` | -0.0792 | [-0.0960, -0.0626] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.5.K3` | -0.0581 | [-0.0687, -0.0472] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.5.K3` | -0.0765 | [-0.0893, -0.0641] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.5.K4` | -0.0557 | [-0.0634, -0.0475] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.5.K4` | -0.0812 | [-0.0904, -0.0717] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.5.K5` | -0.0576 | [-0.0643, -0.0513] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.5.K5` | -0.0830 | [-0.0910, -0.0752] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.5.K6` | -0.0503 | [-0.0553, -0.0453] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.5.K6` | -0.0805 | [-0.0869, -0.0744] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.5.K7` | -0.0536 | [-0.0584, -0.0491] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.5.K7` | -0.0874 | [-0.0935, -0.0818] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.5.K8` | -0.0476 | [-0.0529, -0.0425] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.5.K8` | -0.0799 | [-0.0867, -0.0735] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.5.K9` | -0.0470 | [-0.0514, -0.0423] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.5.K9` | -0.0819 | [-0.0885, -0.0752] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.5.K10` | -0.0514 | [-0.0559, -0.0472] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.5.K10` | -0.0868 | [-0.0921, -0.0818] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.1.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.1.K2` | 0.0490 | [0.0405, 0.0577] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.1.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.1.K3` | 0.0517 | [0.0427, 0.0603] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.1.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.1.K4` | 0.0535 | [0.0439, 0.0623] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.1.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.1.K5` | 0.0578 | [0.0473, 0.0678] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.1.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.1.K6` | 0.0656 | [0.0555, 0.0758] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.1.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.1.K7` | 0.0730 | [0.0642, 0.0814] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.1.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.1.K8` | 0.0708 | [0.0608, 0.0809] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.1.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.1.K9` | 0.0682 | [0.0606, 0.0765] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.rf.dir0.1.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.rf.dir0.1.K10` | 0.0730 | [0.0652, 0.0809] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.1.K2` | -0.0617 | [-0.0756, -0.0484] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.1.K2` | -0.1137 | [-0.1386, -0.0900] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.1.K3` | -0.0721 | [-0.0816, -0.0625] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.1.K3` | -0.1234 | [-0.1375, -0.1093] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.1.K4` | -0.0679 | [-0.0766, -0.0596] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.1.K4` | -0.1115 | [-0.1236, -0.0995] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.1.K5` | -0.0626 | [-0.0697, -0.0556] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.1.K5` | -0.1168 | [-0.1263, -0.1071] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.1.K6` | -0.0594 | [-0.0648, -0.0538] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.1.K6` | -0.1072 | [-0.1149, -0.0993] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.1.K7` | -0.0527 | [-0.0574, -0.0474] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.1.K7` | -0.1009 | [-0.1084, -0.0927] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.1.K8` | -0.0543 | [-0.0586, -0.0501] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.1.K8` | -0.1003 | [-0.1057, -0.0949] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.1.K9` | -0.0514 | [-0.0554, -0.0470] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.1.K9` | -0.0977 | [-0.1037, -0.0917] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.rf.dir0.1.K10` | -0.0518 | [-0.0558, -0.0479] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.rf.dir0.1.K10` | -0.0969 | [-0.1032, -0.0909] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.equal.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.equal.K2` | 0.0312 | [0.0288, 0.0334] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.equal.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.equal.K3` | 0.0458 | [0.0427, 0.0489] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.equal.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.equal.K4` | 0.0594 | [0.0558, 0.0627] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.equal.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.equal.K5` | 0.0676 | [0.0633, 0.0717] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.equal.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.equal.K6` | 0.0749 | [0.0704, 0.0793] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.equal.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.equal.K7` | 0.0790 | [0.0746, 0.0837] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.equal.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.equal.K8` | 0.0817 | [0.0771, 0.0861] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.equal.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.equal.K9` | 0.0868 | [0.0821, 0.0918] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.equal.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.equal.K10` | 0.0881 | [0.0833, 0.0930] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.equal.K2` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.equal.K2` | -0.0285 | [-0.0302, -0.0268] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.equal.K3` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.equal.K3` | -0.0423 | [-0.0441, -0.0403] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.equal.K4` | -0.0000 | [-0.0000, -0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.equal.K4` | -0.0541 | [-0.0560, -0.0522] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.equal.K5` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.equal.K5` | -0.0603 | [-0.0623, -0.0584] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.equal.K6` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.equal.K6` | -0.0655 | [-0.0677, -0.0631] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.equal.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.equal.K7` | -0.0704 | [-0.0727, -0.0683] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.equal.K8` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.equal.K8` | -0.0730 | [-0.0751, -0.0708] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.equal.K9` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.equal.K9` | -0.0758 | [-0.0782, -0.0735] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.equal.K10` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.equal.K10` | -0.0781 | [-0.0799, -0.0762] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.5.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.5.K2` | 0.0470 | [0.0419, 0.0530] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.5.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.5.K3` | 0.0634 | [0.0584, 0.0686] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.5.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.5.K4` | 0.0813 | [0.0753, 0.0868] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.5.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.5.K5` | 0.0916 | [0.0853, 0.0982] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.5.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.5.K6` | 0.0945 | [0.0884, 0.1009] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.5.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.5.K7` | 0.1030 | [0.0972, 0.1089] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.5.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.5.K8` | 0.1032 | [0.0957, 0.1107] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.5.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.5.K9` | 0.1036 | [0.0966, 0.1113] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.5.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.5.K10` | 0.1056 | [0.0980, 0.1125] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.5.K2` | -0.0508 | [-0.0612, -0.0406] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.5.K2` | -0.0919 | [-0.1088, -0.0755] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.5.K3` | -0.0462 | [-0.0550, -0.0374] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.5.K3` | -0.1008 | [-0.1130, -0.0881] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.5.K4` | -0.0487 | [-0.0558, -0.0412] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.5.K4` | -0.1171 | [-0.1277, -0.1062] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.5.K5` | -0.0472 | [-0.0530, -0.0416] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.5.K5` | -0.1225 | [-0.1320, -0.1129] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.5.K6` | -0.0427 | [-0.0471, -0.0382] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.5.K6` | -0.1241 | [-0.1315, -0.1168] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.5.K7` | -0.0459 | [-0.0500, -0.0418] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.5.K7` | -0.1372 | [-0.1434, -0.1312] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.5.K8` | -0.0408 | [-0.0458, -0.0358] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.5.K8` | -0.1278 | [-0.1344, -0.1215] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.5.K9` | -0.0410 | [-0.0451, -0.0369] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.5.K9` | -0.1317 | [-0.1378, -0.1255] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.5.K10` | -0.0441 | [-0.0477, -0.0407] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.5.K10` | -0.1350 | [-0.1418, -0.1278] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.1.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.1.K2` | 0.0605 | [0.0511, 0.0705] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.1.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.1.K3` | 0.0778 | [0.0700, 0.0854] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.1.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.1.K4` | 0.0969 | [0.0877, 0.1061] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.1.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.1.K5` | 0.0946 | [0.0860, 0.1027] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.1.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.1.K6` | 0.1061 | [0.0976, 0.1145] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.1.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.1.K7` | 0.1133 | [0.1055, 0.1216] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.1.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.1.K8` | 0.1105 | [0.1032, 0.1179] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.1.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.1.K9` | 0.1101 | [0.1017, 0.1186] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.diabetic.xgb.dir0.1.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.diabetic.xgb.dir0.1.K10` | 0.1073 | [0.0989, 0.1159] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.1.K2` | -0.0668 | [-0.0824, -0.0523] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.1.K2` | -0.1133 | [-0.1370, -0.0904] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.1.K3` | -0.0705 | [-0.0802, -0.0608] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.1.K3` | -0.1352 | [-0.1474, -0.1225] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.1.K4` | -0.0669 | [-0.0758, -0.0586] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.1.K4` | -0.1269 | [-0.1367, -0.1165] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.1.K5` | -0.0638 | [-0.0703, -0.0569] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.1.K5` | -0.1330 | [-0.1441, -0.1215] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.1.K6` | -0.0601 | [-0.0653, -0.0546] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.1.K6` | -0.1286 | [-0.1377, -0.1197] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.1.K7` | -0.0532 | [-0.0586, -0.0473] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.1.K7` | -0.1279 | [-0.1362, -0.1196] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.1.K8` | -0.0519 | [-0.0565, -0.0476] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.1.K8` | -0.1302 | [-0.1382, -0.1226] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.1.K9` | -0.0506 | [-0.0551, -0.0459] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.1.K9` | -0.1249 | [-0.1324, -0.1170] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.diabetic.xgb.dir0.1.K10` | -0.0501 | [-0.0539, -0.0463] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.diabetic.xgb.dir0.1.K10` | -0.1262 | [-0.1330, -0.1196] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.equal.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.equal.K2` | 0.0037 | [0.0034, 0.0039] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.equal.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.equal.K3` | 0.0053 | [0.0050, 0.0057] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.equal.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.equal.K4` | 0.0069 | [0.0065, 0.0073] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.equal.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.equal.K5` | 0.0071 | [0.0067, 0.0075] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.equal.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.equal.K6` | 0.0078 | [0.0074, 0.0082] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.equal.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.equal.K7` | 0.0083 | [0.0079, 0.0087] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.equal.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.equal.K8` | 0.0093 | [0.0089, 0.0097] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.equal.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.equal.K9` | 0.0094 | [0.0089, 0.0098] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.equal.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.equal.K10` | 0.0091 | [0.0086, 0.0096] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.equal.K2` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.equal.K2` | -0.0002 | [-0.0002, -0.0002] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.equal.K3` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.equal.K3` | -0.0003 | [-0.0003, -0.0003] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.equal.K4` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.equal.K4` | -0.0004 | [-0.0004, -0.0004] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.equal.K5` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.equal.K5` | -0.0004 | [-0.0004, -0.0004] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.equal.K6` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.equal.K6` | -0.0004 | [-0.0005, -0.0004] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.equal.K7` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.equal.K7` | -0.0005 | [-0.0005, -0.0004] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.equal.K8` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.equal.K8` | -0.0005 | [-0.0005, -0.0005] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.equal.K9` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.equal.K9` | -0.0005 | [-0.0006, -0.0005] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.equal.K10` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.equal.K10` | -0.0005 | [-0.0005, -0.0005] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.5.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.5.K2` | 0.0316 | [0.0225, 0.0421] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.5.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.5.K3` | 0.0515 | [0.0425, 0.0610] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.5.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.5.K4` | 0.0565 | [0.0477, 0.0656] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.5.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.5.K5` | 0.0553 | [0.0468, 0.0639] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.5.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.5.K6` | 0.0603 | [0.0528, 0.0682] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.5.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.5.K7` | 0.0592 | [0.0535, 0.0650] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.5.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.5.K8` | 0.0641 | [0.0575, 0.0709] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.5.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.5.K9` | 0.0647 | [0.0585, 0.0706] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.5.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.5.K10` | 0.0655 | [0.0592, 0.0723] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.5.K2` | -0.0485 | [-0.0723, -0.0256] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.5.K2` | -0.0323 | [-0.0486, -0.0174] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.5.K3` | -0.0687 | [-0.0930, -0.0469] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.5.K3` | -0.0481 | [-0.0632, -0.0346] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.5.K4` | -0.0426 | [-0.0570, -0.0303] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.5.K4` | -0.0372 | [-0.0494, -0.0268] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.5.K5` | -0.0466 | [-0.0613, -0.0341] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.5.K5` | -0.0435 | [-0.0569, -0.0312] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.5.K6` | -0.0482 | [-0.0591, -0.0383] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.5.K6` | -0.0422 | [-0.0514, -0.0334] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.5.K7` | -0.0498 | [-0.0614, -0.0391] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.5.K7` | -0.0412 | [-0.0503, -0.0330] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.5.K8` | -0.0739 | [-0.0870, -0.0614] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.5.K8` | -0.0644 | [-0.0750, -0.0544] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.5.K9` | -0.0600 | [-0.0744, -0.0465] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.5.K9` | -0.0533 | [-0.0650, -0.0422] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.5.K10` | -0.0595 | [-0.0713, -0.0482] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.5.K10` | -0.0514 | [-0.0603, -0.0427] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.1.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.1.K2` | 0.0563 | [0.0422, 0.0703] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.1.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.1.K3` | 0.0876 | [0.0709, 0.1043] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.1.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.1.K4` | 0.1143 | [0.1034, 0.1250] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.1.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.1.K5` | 0.1069 | [0.0980, 0.1156] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.1.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.1.K6` | 0.1092 | [0.1003, 0.1182] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.1.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.1.K7` | 0.1199 | [0.1123, 0.1276] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.1.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.1.K8` | 0.1130 | [0.1042, 0.1211] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.1.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.1.K9` | 0.1151 | [0.1083, 0.1217] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.rf.dir0.1.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.rf.dir0.1.K10` | 0.1151 | [0.1083, 0.1218] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.1.K2` | -0.0910 | [-0.1193, -0.0666] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.1.K2` | -0.1169 | [-0.1461, -0.0890] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.1.K3` | -0.1051 | [-0.1364, -0.0772] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.1.K3` | -0.1054 | [-0.1286, -0.0830] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.1.K4` | -0.1360 | [-0.1595, -0.1136] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.1.K4` | -0.1381 | [-0.1535, -0.1210] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.1.K5` | -0.1581 | [-0.1786, -0.1384] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.1.K5` | -0.1406 | [-0.1528, -0.1276] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.1.K6` | -0.1336 | [-0.1554, -0.1138] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.1.K6` | -0.1226 | [-0.1371, -0.1078] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.1.K7` | -0.1340 | [-0.1509, -0.1167] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.1.K7` | -0.1364 | [-0.1481, -0.1242] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.1.K8` | -0.1375 | [-0.1558, -0.1195] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.1.K8` | -0.1274 | [-0.1372, -0.1164] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.1.K9` | -0.1538 | [-0.1689, -0.1388] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.1.K9` | -0.1258 | [-0.1356, -0.1152] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.rf.dir0.1.K10` | -0.1488 | [-0.1672, -0.1321] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.rf.dir0.1.K10` | -0.1214 | [-0.1322, -0.1108] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.equal.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.equal.K2` | 0.0105 | [0.0104, 0.0107] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.equal.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.equal.K3` | 0.0176 | [0.0175, 0.0178] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.equal.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.equal.K4` | 0.0233 | [0.0231, 0.0235] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.equal.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.equal.K5` | 0.0275 | [0.0274, 0.0277] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.equal.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.equal.K6` | 0.0315 | [0.0313, 0.0317] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.equal.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.equal.K7` | 0.0345 | [0.0343, 0.0347] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.equal.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.equal.K8` | 0.0371 | [0.0369, 0.0373] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.equal.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.equal.K9` | 0.0396 | [0.0394, 0.0398] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.equal.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.equal.K10` | 0.0416 | [0.0414, 0.0419] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.equal.K2` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.equal.K2` | -0.0012 | [-0.0012, -0.0012] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.equal.K3` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.equal.K3` | -0.0019 | [-0.0019, -0.0019] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.equal.K4` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.equal.K4` | -0.0024 | [-0.0024, -0.0023] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.equal.K5` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.equal.K5` | -0.0027 | [-0.0027, -0.0027] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.equal.K6` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.equal.K6` | -0.0030 | [-0.0030, -0.0030] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.equal.K7` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.equal.K7` | -0.0032 | [-0.0032, -0.0031] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.equal.K8` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.equal.K8` | -0.0033 | [-0.0033, -0.0033] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.equal.K9` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.equal.K9` | -0.0034 | [-0.0035, -0.0034] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.equal.K10` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.equal.K10` | -0.0035 | [-0.0036, -0.0035] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.5.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.5.K2` | 0.0167 | [0.0145, 0.0191] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.5.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.5.K3` | 0.0315 | [0.0286, 0.0348] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.5.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.5.K4` | 0.0402 | [0.0374, 0.0430] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.5.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.5.K5` | 0.0489 | [0.0450, 0.0531] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.5.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.5.K6` | 0.0514 | [0.0484, 0.0548] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.5.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.5.K7` | 0.0560 | [0.0529, 0.0592] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.5.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.5.K8` | 0.0613 | [0.0579, 0.0649] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.5.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.5.K9` | 0.0636 | [0.0607, 0.0665] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.5.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.5.K10` | 0.0668 | [0.0638, 0.0699] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.5.K2` | -0.0495 | [-0.0736, -0.0270] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.5.K2` | -0.0393 | [-0.0629, -0.0196] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.5.K3` | -0.0691 | [-0.0925, -0.0477] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.5.K3` | -0.0609 | [-0.0790, -0.0436] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.5.K4` | -0.0466 | [-0.0608, -0.0336] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.5.K4` | -0.0515 | [-0.0683, -0.0368] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.5.K5` | -0.0533 | [-0.0696, -0.0386] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.5.K5` | -0.0596 | [-0.0785, -0.0427] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.5.K6` | -0.0497 | [-0.0604, -0.0398] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.5.K6` | -0.0590 | [-0.0718, -0.0470] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.5.K7` | -0.0514 | [-0.0631, -0.0406] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.5.K7` | -0.0604 | [-0.0746, -0.0476] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.5.K8` | -0.0763 | [-0.0897, -0.0638] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.5.K8` | -0.0935 | [-0.1100, -0.0783] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.5.K9` | -0.0622 | [-0.0768, -0.0485] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.5.K9` | -0.0778 | [-0.0961, -0.0610] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.5.K10` | -0.0599 | [-0.0712, -0.0493] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.5.K10` | -0.0752 | [-0.0895, -0.0618] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.1.K2` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.1.K2` | 0.0381 | [0.0317, 0.0449] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.1.K3` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.1.K3` | 0.0683 | [0.0625, 0.0742] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.1.K4` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.1.K4` | 0.0788 | [0.0729, 0.0846] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.1.K5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.1.K5` | 0.0912 | [0.0869, 0.0953] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.1.K6` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.1.K6` | 0.1008 | [0.0970, 0.1047] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.1.K7` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.1.K7` | 0.1064 | [0.1016, 0.1113] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.1.K8` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.1.K8` | 0.1118 | [0.1078, 0.1162] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.1.K9` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.1.K9` | 0.1149 | [0.1106, 0.1192] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L1.heart.xgb.dir0.1.K10` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.auroc.L3.heart.xgb.dir0.1.K10` | 0.1178 | [0.1130, 0.1225] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.1.K2` | -0.1707 | [-0.2112, -0.1320] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.1.K2` | -0.1058 | [-0.1518, -0.0636] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.1.K3` | -0.1569 | [-0.1934, -0.1219] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.1.K3` | -0.1322 | [-0.1687, -0.0987] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.1.K4` | -0.2015 | [-0.2291, -0.1740] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.1.K4` | -0.1975 | [-0.2225, -0.1719] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.1.K5` | -0.2205 | [-0.2425, -0.1981] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.1.K5` | -0.2191 | [-0.2425, -0.1934] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.1.K6` | -0.1985 | [-0.2287, -0.1702] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.1.K6` | -0.1767 | [-0.2031, -0.1495] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.1.K7` | -0.2029 | [-0.2271, -0.1789] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.1.K7` | -0.1982 | [-0.2209, -0.1730] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.1.K8` | -0.1912 | [-0.2128, -0.1688] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.1.K8` | -0.1989 | [-0.2168, -0.1796] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.1.K9` | -0.2012 | [-0.2208, -0.1820] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.1.K9` | -0.2001 | [-0.2196, -0.1800] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L1.heart.xgb.dir0.1.K10` | -0.1976 | [-0.2217, -0.1745] | 50 | Mean over draws at this K of the level minus L0. |
| `perK.brier.L3.heart.xgb.dir0.1.K10` | -0.1986 | [-0.2196, -0.1781] | 50 | Mean over draws at this K of the level minus L0. |
| `leaf.rf.central.central.aids.single` | 0.5120 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.rf.central.central.aids.lt5` | 0.8416 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.xgb.central.central.aids.single` | 0.0000 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.xgb.central.central.aids.lt5` | 0.0000 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.rf.site.all.aids.single` | 0.4990 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.rf.site.all.aids.lt5` | 0.8219 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.xgb.site.all.aids.single` | 0.0000 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.xgb.site.all.aids.lt5` | 0.0001 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.rf.site.dir0.1.aids.single` | 0.4881 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.rf.site.dir0.1.aids.lt5` | 0.7947 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.xgb.site.dir0.1.aids.single` | 0.0000 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.xgb.site.dir0.1.aids.lt5` | 0.0003 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.rf.site.dir0.5.aids.single` | 0.4971 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.rf.site.dir0.5.aids.lt5` | 0.8143 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.xgb.site.dir0.5.aids.single` | 0.0000 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.xgb.site.dir0.5.aids.lt5` | 0.0000 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.rf.site.equal.aids.single` | 0.5053 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.rf.site.equal.aids.lt5` | 0.8392 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.xgb.site.equal.aids.single` | 0.0000 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.xgb.site.equal.aids.lt5` | 0.0000 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.rf.central.central.diabetic.single` | 0.4220 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.rf.central.central.diabetic.lt5` | 0.7794 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.xgb.central.central.diabetic.single` | 0.0000 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.xgb.central.central.diabetic.lt5` | 0.0000 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.rf.site.all.diabetic.single` | 0.4206 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.rf.site.all.diabetic.lt5` | 0.7679 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.xgb.site.all.diabetic.single` | 0.0000 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.xgb.site.all.diabetic.lt5` | 0.0001 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.rf.site.dir0.1.diabetic.single` | 0.4263 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.rf.site.dir0.1.diabetic.lt5` | 0.7548 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.xgb.site.dir0.1.diabetic.single` | 0.0000 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.xgb.site.dir0.1.diabetic.lt5` | 0.0004 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.rf.site.dir0.5.diabetic.single` | 0.4256 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.rf.site.dir0.5.diabetic.lt5` | 0.7651 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.xgb.site.dir0.5.diabetic.single` | 0.0000 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.xgb.site.dir0.5.diabetic.lt5` | 0.0000 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.rf.site.equal.diabetic.single` | 0.4144 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.rf.site.equal.diabetic.lt5` | 0.7761 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.xgb.site.equal.diabetic.single` | 0.0000 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.xgb.site.equal.diabetic.lt5` | 0.0000 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.rf.central.central.heart.single` | 0.4700 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.rf.central.central.heart.lt5` | 0.7444 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.xgb.central.central.heart.single` | 0.0000 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.xgb.central.central.heart.lt5` | 0.0000 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.rf.site.all.heart.single` | 0.4640 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.rf.site.all.heart.lt5` | 0.7423 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.xgb.site.all.heart.single` | 0.0000 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.xgb.site.all.heart.lt5` | 0.0000 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.rf.site.dir0.1.heart.single` | 0.4642 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.rf.site.dir0.1.heart.lt5` | 0.7439 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.xgb.site.dir0.1.heart.single` | 0.0000 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.xgb.site.dir0.1.heart.lt5` | 0.0000 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.rf.site.dir0.5.heart.single` | 0.4636 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.rf.site.dir0.5.heart.lt5` | 0.7458 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.xgb.site.dir0.5.heart.single` | 0.0000 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.xgb.site.dir0.5.heart.lt5` | 0.0000 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.rf.site.equal.heart.single` | 0.4642 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.rf.site.equal.heart.lt5` | 0.7392 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `leaf.xgb.site.equal.heart.single` | 0.0000 |  |  | Leaves reached by exactly one distinct training row, pooled over trees, sites, cells and draws. |
| `leaf.xgb.site.equal.heart.lt5` | 0.0000 |  |  | Leaves reached by fewer than 5 distinct training rows, pooled the same way. |
| `heart.K5.auroc.source.rf.equal` | 0.8186 | [0.8175, 0.8198] | 50 | AUROC of the mean source model at K = 5. |
| `heart.K5.auroc.central.rf.equal` | 0.8252 | [0.8240, 0.8265] | 50 | AUROC of the centralized model (draws of K = 5 cells). |
| `heart.K5.auroc.source.rf.dir0.5` | 0.7740 | [0.7641, 0.7835] | 50 | AUROC of the mean source model at K = 5. |
| `heart.K5.auroc.central.rf.dir0.5` | 0.8252 | [0.8240, 0.8265] | 50 | AUROC of the centralized model (draws of K = 5 cells). |
| `heart.K5.auroc.source.rf.dir0.1` | 0.6800 | [0.6678, 0.6915] | 50 | AUROC of the mean source model at K = 5. |
| `heart.K5.auroc.central.rf.dir0.1` | 0.8252 | [0.8240, 0.8265] | 50 | AUROC of the centralized model (draws of K = 5 cells). |
| `heart.K5.auroc.source.xgb.equal` | 0.8109 | [0.8099, 0.8118] | 50 | AUROC of the mean source model at K = 5. |
| `heart.K5.auroc.central.xgb.equal` | 0.8365 | [0.8354, 0.8377] | 50 | AUROC of the centralized model (draws of K = 5 cells). |
| `heart.K5.auroc.source.xgb.dir0.5` | 0.7819 | [0.7762, 0.7873] | 50 | AUROC of the mean source model at K = 5. |
| `heart.K5.auroc.central.xgb.dir0.5` | 0.8365 | [0.8354, 0.8377] | 50 | AUROC of the centralized model (draws of K = 5 cells). |
| `heart.K5.auroc.source.xgb.dir0.1` | 0.7157 | [0.7077, 0.7230] | 50 | AUROC of the mean source model at K = 5. |
| `heart.K5.auroc.central.xgb.dir0.1` | 0.8365 | [0.8354, 0.8377] | 50 | AUROC of the centralized model (draws of K = 5 cells). |
| `heart.rf.K10.ppr05.fed_L0.equal` | 0.0002 | [0.0002, 0.0002] | 50 | Share of positive predictions at threshold 0.5 of fed_L0 rf at K = 10. |
| `heart.rf.K10.ppr05.source.equal` | 0.0006 | [0.0006, 0.0006] | 50 | Same for the mean source model. |
| `heart.rf.K10.auroc.fed_L0.equal` | 0.8240 | [0.8228, 0.8252] | 50 | AUROC [A] of fed_L0 rf at K = 10. |
| `heart.rf.K10.auroc.source.equal` | 0.8149 | [0.8137, 0.8161] | 50 | AUROC [B] of the mean source model rf at K = 10. |
| `heart.rf.K10.mcc_pi.diff.equal` | 0.0051 | [0.0043, 0.0058] | 50 | mcc_pi of fed_L0 rf minus the mean source model at K = 10. |
| `heart.rf.K10.ppr05.fed_L0.dir0.5` | 0.0034 | [0.0017, 0.0055] | 50 | Share of positive predictions at threshold 0.5 of fed_L0 rf at K = 10. |
| `heart.rf.K10.ppr05.source.dir0.5` | 0.1167 | [0.0968, 0.1377] | 50 | Same for the mean source model. |
| `heart.rf.K10.auroc.fed_L0.dir0.5` | 0.8300 | [0.8284, 0.8315] | 50 | AUROC [A] of fed_L0 rf at K = 10. |
| `heart.rf.K10.auroc.source.dir0.5` | 0.7645 | [0.7571, 0.7715] | 50 | AUROC [B] of the mean source model rf at K = 10. |
| `heart.rf.K10.mcc_pi.diff.dir0.5` | -0.0391 | [-0.0548, -0.0241] | 50 | mcc_pi of fed_L0 rf minus the mean source model at K = 10. |
| `heart.rf.K10.ppr05.fed_L0.dir0.1` | 0.1153 | [0.0612, 0.1794] | 50 | Share of positive predictions at threshold 0.5 of fed_L0 rf at K = 10. |
| `heart.rf.K10.ppr05.source.dir0.1` | 0.3203 | [0.2860, 0.3575] | 50 | Same for the mean source model. |
| `heart.rf.K10.auroc.fed_L0.dir0.1` | 0.8005 | [0.7944, 0.8063] | 50 | AUROC [A] of fed_L0 rf at K = 10. |
| `heart.rf.K10.auroc.source.dir0.1` | 0.6853 | [0.6767, 0.6941] | 50 | AUROC [B] of the mean source model rf at K = 10. |
| `heart.rf.K10.mcc_pi.diff.dir0.1` | -0.0522 | [-0.0591, -0.0459] | 50 | mcc_pi of fed_L0 rf minus the mean source model at K = 10. |
| `calib.citl_diff.fed_L0.xgb.equal.aids` | -0.1099 | [-0.1125, -0.1075] | 50 | Mean over draws, K averaged. |
| `calib.citl_int.fed_L0.xgb.equal.aids` | 1.0685 | [1.0410, 1.0997] | 50 | Mean over draws, K averaged. |
| `calib.citl_diff.fed_L0.xgb.equal.diabetic` | -0.0114 | [-0.0172, -0.0055] | 50 | Mean over draws, K averaged. |
| `calib.citl_int.fed_L0.xgb.equal.diabetic` | 0.0756 | [0.0372, 0.1151] | 50 | Mean over draws, K averaged. |
| `calib.citl_diff.fed_L0.xgb.equal.heart` | -0.0147 | [-0.0148, -0.0146] | 50 | Mean over draws, K averaged. |
| `calib.citl_int.fed_L0.xgb.equal.heart` | 0.3981 | [0.3949, 0.4013] | 50 | Mean over draws, K averaged. |
| `e5.auroc.fed_L0.aids.rf.equal.equal_oob` | -0.0009 | [-0.0023, 0.0005] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.rf.equal.equal_oob` | 0.0001 | [-0.0001, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.rf.equal.equal_oob` | -0.0002 | [-0.0040, 0.0037] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.rf.equal.positives_oob` | -0.0004 | [-0.0019, 0.0010] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.rf.equal.positives_oob` | -0.0000 | [-0.0003, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.rf.equal.positives_oob` | 0.0026 | [-0.0016, 0.0067] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.rf.equal.positives_uniform` | -0.0013 | [-0.0027, 0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.rf.equal.positives_uniform` | 0.0002 | [-0.0001, 0.0004] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.rf.equal.positives_uniform` | -0.0025 | [-0.0060, 0.0009] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.rf.equal.size_oob` | -0.0001 | [-0.0015, 0.0013] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.rf.equal.size_oob` | -0.0000 | [-0.0003, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.rf.equal.size_oob` | 0.0001 | [-0.0036, 0.0037] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.rf.equal.size_uniform` | -0.0003 | [-0.0018, 0.0013] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.rf.equal.size_uniform` | 0.0001 | [-0.0002, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.rf.equal.size_uniform` | 0.0019 | [-0.0019, 0.0054] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.rf.equal.equal_oob` | -0.0009 | [-0.0023, 0.0005] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.rf.equal.equal_oob` | 0.0001 | [-0.0001, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.rf.equal.equal_oob` | -0.0002 | [-0.0040, 0.0037] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.rf.equal.positives_oob` | -0.0004 | [-0.0019, 0.0011] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.rf.equal.positives_oob` | -0.0000 | [-0.0003, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.rf.equal.positives_oob` | 0.0026 | [-0.0016, 0.0067] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.rf.equal.positives_uniform` | -0.0013 | [-0.0028, 0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.rf.equal.positives_uniform` | 0.0002 | [-0.0001, 0.0004] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.rf.equal.positives_uniform` | -0.0025 | [-0.0060, 0.0009] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.rf.equal.size_oob` | -0.0003 | [-0.0016, 0.0011] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.rf.equal.size_oob` | -0.0000 | [-0.0003, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.rf.equal.size_oob` | 0.0001 | [-0.0036, 0.0037] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.rf.equal.size_uniform` | -0.0003 | [-0.0018, 0.0012] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.rf.equal.size_uniform` | 0.0001 | [-0.0002, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.rf.equal.size_uniform` | 0.0019 | [-0.0019, 0.0054] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.rf.equal.equal_oob` | -0.0015 | [-0.0057, 0.0021] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.rf.equal.equal_oob` | -0.0002 | [-0.0005, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.rf.equal.equal_oob` | -0.0002 | [-0.0053, 0.0040] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.rf.equal.positives_oob` | -0.0028 | [-0.0073, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.rf.equal.positives_oob` | -0.0001 | [-0.0003, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.rf.equal.positives_oob` | -0.0017 | [-0.0088, 0.0041] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.rf.equal.positives_uniform` | -0.0003 | [-0.0022, 0.0021] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.rf.equal.positives_uniform` | -0.0001 | [-0.0004, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.rf.equal.positives_uniform` | 0.0009 | [-0.0030, 0.0049] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.rf.equal.size_oob` | 0.0001 | [-0.0029, 0.0032] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.rf.equal.size_oob` | -0.0002 | [-0.0004, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.rf.equal.size_oob` | 0.0004 | [-0.0050, 0.0054] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.rf.equal.size_uniform` | -0.0019 | [-0.0047, 0.0005] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.rf.equal.size_uniform` | -0.0001 | [-0.0004, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.rf.equal.size_uniform` | -0.0020 | [-0.0068, 0.0022] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.rf.dir0.5.equal_oob` | 0.0010 | [-0.0007, 0.0027] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.rf.dir0.5.equal_oob` | -0.0002 | [-0.0004, 0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.rf.dir0.5.equal_oob` | 0.0033 | [-0.0013, 0.0076] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.rf.dir0.5.positives_oob` | -0.0012 | [-0.0037, 0.0012] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.rf.dir0.5.positives_oob` | 0.0603 | [0.0529, 0.0679] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.rf.dir0.5.positives_oob` | -0.0404 | [-0.0472, -0.0335] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.rf.dir0.5.positives_uniform` | -0.0016 | [-0.0040, 0.0009] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.rf.dir0.5.positives_uniform` | 0.0604 | [0.0531, 0.0680] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.rf.dir0.5.positives_uniform` | -0.0396 | [-0.0464, -0.0326] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.rf.dir0.5.size_oob` | 0.0023 | [-0.0005, 0.0050] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.rf.dir0.5.size_oob` | -0.0168 | [-0.0190, -0.0147] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.rf.dir0.5.size_oob` | 0.0911 | [0.0820, 0.1002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.rf.dir0.5.size_uniform` | 0.0018 | [-0.0005, 0.0041] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.rf.dir0.5.size_uniform` | -0.0168 | [-0.0189, -0.0148] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.rf.dir0.5.size_uniform` | 0.0918 | [0.0825, 0.1012] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.rf.dir0.5.equal_oob` | -0.0005 | [-0.0024, 0.0015] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.rf.dir0.5.equal_oob` | 0.0001 | [-0.0002, 0.0005] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.rf.dir0.5.equal_oob` | -0.0010 | [-0.0054, 0.0032] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.rf.dir0.5.positives_oob` | -0.0009 | [-0.0037, 0.0017] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.rf.dir0.5.positives_oob` | 0.0059 | [0.0041, 0.0078] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.rf.dir0.5.positives_oob` | 0.0027 | [-0.0013, 0.0068] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.rf.dir0.5.positives_uniform` | -0.0016 | [-0.0045, 0.0013] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.rf.dir0.5.positives_uniform` | 0.0061 | [0.0043, 0.0081] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.rf.dir0.5.positives_uniform` | 0.0035 | [-0.0013, 0.0080] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.rf.dir0.5.size_oob` | 0.0016 | [-0.0011, 0.0043] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.rf.dir0.5.size_oob` | -0.0004 | [-0.0010, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.rf.dir0.5.size_oob` | 0.0064 | [0.0019, 0.0114] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.rf.dir0.5.size_uniform` | 0.0004 | [-0.0023, 0.0031] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.rf.dir0.5.size_uniform` | -0.0000 | [-0.0007, 0.0006] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.rf.dir0.5.size_uniform` | 0.0043 | [-0.0012, 0.0098] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.rf.dir0.5.equal_oob` | -0.0012 | [-0.0056, 0.0019] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.rf.dir0.5.equal_oob` | -0.0002 | [-0.0005, 0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.rf.dir0.5.equal_oob` | -0.0029 | [-0.0085, 0.0025] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.rf.dir0.5.positives_oob` | -0.0015 | [-0.0051, 0.0022] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.rf.dir0.5.positives_oob` | 0.0001 | [-0.0003, 0.0005] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.rf.dir0.5.positives_oob` | -0.0054 | [-0.0116, 0.0013] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.rf.dir0.5.positives_uniform` | -0.0025 | [-0.0065, 0.0015] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.rf.dir0.5.positives_uniform` | 0.0001 | [-0.0002, 0.0005] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.rf.dir0.5.positives_uniform` | -0.0018 | [-0.0078, 0.0041] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.rf.dir0.5.size_oob` | 0.0049 | [0.0016, 0.0084] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.rf.dir0.5.size_oob` | -0.0002 | [-0.0006, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.rf.dir0.5.size_oob` | 0.0037 | [-0.0014, 0.0087] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.rf.dir0.5.size_uniform` | 0.0009 | [-0.0033, 0.0046] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.rf.dir0.5.size_uniform` | -0.0002 | [-0.0006, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.rf.dir0.5.size_uniform` | 0.0022 | [-0.0040, 0.0077] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.rf.dir0.1.equal_oob` | -0.0002 | [-0.0028, 0.0022] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.rf.dir0.1.equal_oob` | -0.0001 | [-0.0006, 0.0005] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.rf.dir0.1.equal_oob` | 0.0001 | [-0.0032, 0.0032] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.rf.dir0.1.positives_oob` | -0.0122 | [-0.0167, -0.0080] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.rf.dir0.1.positives_oob` | 0.2125 | [0.1980, 0.2272] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.rf.dir0.1.positives_oob` | 0.0056 | [0.0004, 0.0108] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.rf.dir0.1.positives_uniform` | -0.0117 | [-0.0158, -0.0077] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.rf.dir0.1.positives_uniform` | 0.2123 | [0.1977, 0.2269] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.rf.dir0.1.positives_uniform` | 0.0063 | [0.0007, 0.0121] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.rf.dir0.1.size_oob` | 0.0019 | [-0.0017, 0.0055] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.rf.dir0.1.size_oob` | -0.0457 | [-0.0491, -0.0425] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.rf.dir0.1.size_oob` | 0.1231 | [0.1160, 0.1307] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.rf.dir0.1.size_uniform` | 0.0003 | [-0.0037, 0.0042] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.rf.dir0.1.size_uniform` | -0.0456 | [-0.0490, -0.0424] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.rf.dir0.1.size_uniform` | 0.1240 | [0.1165, 0.1312] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.rf.dir0.1.equal_oob` | 0.0003 | [-0.0023, 0.0029] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.rf.dir0.1.equal_oob` | 0.0002 | [-0.0004, 0.0007] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.rf.dir0.1.equal_oob` | 0.0015 | [-0.0034, 0.0067] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.rf.dir0.1.positives_oob` | -0.0189 | [-0.0237, -0.0145] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.rf.dir0.1.positives_oob` | 0.0810 | [0.0752, 0.0868] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.rf.dir0.1.positives_oob` | -0.0099 | [-0.0165, -0.0036] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.rf.dir0.1.positives_uniform` | -0.0182 | [-0.0228, -0.0137] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.rf.dir0.1.positives_uniform` | 0.0806 | [0.0749, 0.0865] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.rf.dir0.1.positives_uniform` | -0.0082 | [-0.0145, -0.0021] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.rf.dir0.1.size_oob` | 0.0003 | [-0.0030, 0.0036] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.rf.dir0.1.size_oob` | 0.0006 | [-0.0009, 0.0020] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.rf.dir0.1.size_oob` | 0.0214 | [0.0153, 0.0276] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.rf.dir0.1.size_uniform` | -0.0009 | [-0.0043, 0.0024] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.rf.dir0.1.size_uniform` | 0.0007 | [-0.0007, 0.0020] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.rf.dir0.1.size_uniform` | 0.0229 | [0.0174, 0.0282] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.rf.dir0.1.equal_oob` | 0.0041 | [-0.0012, 0.0095] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.rf.dir0.1.equal_oob` | -0.0001 | [-0.0005, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.rf.dir0.1.equal_oob` | 0.0023 | [-0.0046, 0.0094] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.rf.dir0.1.positives_oob` | -0.0072 | [-0.0132, -0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.rf.dir0.1.positives_oob` | 0.0009 | [0.0004, 0.0015] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.rf.dir0.1.positives_oob` | -0.0118 | [-0.0198, -0.0034] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.rf.dir0.1.positives_uniform` | -0.0090 | [-0.0155, -0.0025] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.rf.dir0.1.positives_uniform` | 0.0010 | [0.0003, 0.0016] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.rf.dir0.1.positives_uniform` | -0.0147 | [-0.0235, -0.0057] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.rf.dir0.1.size_oob` | 0.0052 | [-0.0006, 0.0112] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.rf.dir0.1.size_oob` | 0.0001 | [-0.0005, 0.0006] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.rf.dir0.1.size_oob` | 0.0038 | [-0.0046, 0.0118] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.rf.dir0.1.size_uniform` | 0.0046 | [-0.0018, 0.0114] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.rf.dir0.1.size_uniform` | 0.0001 | [-0.0005, 0.0007] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.rf.dir0.1.size_uniform` | 0.0061 | [-0.0033, 0.0153] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.xgb.equal.positives_na` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.xgb.equal.positives_na` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.xgb.equal.positives_na` | -0.0000 | [-0.0003, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.xgb.equal.size_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.xgb.equal.size_na` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.xgb.equal.size_na` | 0.0001 | [-0.0001, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.xgb.equal.positives_na` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.xgb.equal.positives_na` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.xgb.equal.positives_na` | -0.0000 | [-0.0003, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.xgb.equal.size_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.xgb.equal.size_na` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.xgb.equal.size_na` | 0.0001 | [-0.0002, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.xgb.equal.positives_na` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.xgb.equal.positives_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.xgb.equal.positives_na` | 0.0001 | [-0.0002, 0.0004] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.xgb.equal.size_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.xgb.equal.size_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.xgb.equal.size_na` | 0.0001 | [-0.0000, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.xgb.dir0.5.positives_na` | -0.0110 | [-0.0130, -0.0091] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.xgb.dir0.5.positives_na` | 0.0990 | [0.0869, 0.1112] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.xgb.dir0.5.positives_na` | -0.0536 | [-0.0623, -0.0438] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.xgb.dir0.5.size_na` | -0.0072 | [-0.0091, -0.0053] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.xgb.dir0.5.size_na` | 0.0070 | [0.0048, 0.0089] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.xgb.dir0.5.size_na` | -0.0228 | [-0.0296, -0.0159] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.xgb.dir0.5.positives_na` | -0.0110 | [-0.0130, -0.0091] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.xgb.dir0.5.positives_na` | 0.0256 | [0.0222, 0.0291] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.xgb.dir0.5.positives_na` | -0.0233 | [-0.0299, -0.0169] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.xgb.dir0.5.size_na` | -0.0072 | [-0.0091, -0.0053] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.xgb.dir0.5.size_na` | 0.0072 | [0.0065, 0.0078] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.xgb.dir0.5.size_na` | -0.0124 | [-0.0175, -0.0073] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.xgb.dir0.5.positives_na` | -0.0130 | [-0.0162, -0.0101] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.xgb.dir0.5.positives_na` | 0.0012 | [0.0009, 0.0015] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.xgb.dir0.5.positives_na` | -0.0165 | [-0.0222, -0.0113] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.xgb.dir0.5.size_na` | -0.0103 | [-0.0145, -0.0069] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.xgb.dir0.5.size_na` | 0.0011 | [0.0008, 0.0014] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.xgb.dir0.5.size_na` | -0.0131 | [-0.0209, -0.0066] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.xgb.dir0.1.positives_na` | -0.0306 | [-0.0348, -0.0265] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.xgb.dir0.1.positives_na` | 0.3046 | [0.2871, 0.3220] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.xgb.dir0.1.positives_na` | -0.0680 | [-0.0767, -0.0597] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.aids.xgb.dir0.1.size_na` | -0.0160 | [-0.0195, -0.0124] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.aids.xgb.dir0.1.size_na` | -0.0003 | [-0.0051, 0.0044] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.aids.xgb.dir0.1.size_na` | -0.0251 | [-0.0344, -0.0153] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.xgb.dir0.1.positives_na` | -0.0306 | [-0.0348, -0.0265] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.xgb.dir0.1.positives_na` | 0.0999 | [0.0934, 0.1068] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.xgb.dir0.1.positives_na` | -0.0529 | [-0.0606, -0.0461] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.aids.xgb.dir0.1.size_na` | -0.0160 | [-0.0195, -0.0124] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.aids.xgb.dir0.1.size_na` | 0.0158 | [0.0150, 0.0167] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.aids.xgb.dir0.1.size_na` | -0.0265 | [-0.0326, -0.0199] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.xgb.dir0.1.positives_na` | -0.0331 | [-0.0415, -0.0251] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.xgb.dir0.1.positives_na` | 0.0027 | [0.0021, 0.0034] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.xgb.dir0.1.positives_na` | -0.0452 | [-0.0561, -0.0342] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.aids.xgb.dir0.1.size_na` | -0.0160 | [-0.0234, -0.0089] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.aids.xgb.dir0.1.size_na` | 0.0017 | [0.0012, 0.0021] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.aids.xgb.dir0.1.size_na` | -0.0198 | [-0.0290, -0.0104] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.rf.equal.equal_oob` | 0.0008 | [-0.0008, 0.0023] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.rf.equal.equal_oob` | -0.0001 | [-0.0005, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.rf.equal.equal_oob` | 0.0023 | [-0.0025, 0.0068] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.rf.equal.positives_oob` | 0.0010 | [-0.0006, 0.0025] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.rf.equal.positives_oob` | -0.0003 | [-0.0007, 0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.rf.equal.positives_oob` | 0.0016 | [-0.0033, 0.0060] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.rf.equal.positives_uniform` | 0.0005 | [-0.0011, 0.0020] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.rf.equal.positives_uniform` | -0.0001 | [-0.0005, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.rf.equal.positives_uniform` | 0.0031 | [-0.0015, 0.0077] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.rf.equal.size_oob` | 0.0014 | [-0.0002, 0.0030] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.rf.equal.size_oob` | -0.0004 | [-0.0008, -0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.rf.equal.size_oob` | 0.0021 | [-0.0028, 0.0069] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.rf.equal.size_uniform` | 0.0011 | [-0.0007, 0.0029] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.rf.equal.size_uniform` | -0.0002 | [-0.0006, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.rf.equal.size_uniform` | 0.0033 | [-0.0011, 0.0077] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.rf.equal.equal_oob` | 0.0007 | [-0.0010, 0.0022] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.rf.equal.equal_oob` | -0.0001 | [-0.0005, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.rf.equal.equal_oob` | 0.0023 | [-0.0025, 0.0068] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.rf.equal.positives_oob` | 0.0010 | [-0.0006, 0.0025] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.rf.equal.positives_oob` | -0.0003 | [-0.0007, 0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.rf.equal.positives_oob` | 0.0016 | [-0.0033, 0.0060] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.rf.equal.positives_uniform` | 0.0003 | [-0.0013, 0.0019] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.rf.equal.positives_uniform` | -0.0001 | [-0.0005, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.rf.equal.positives_uniform` | 0.0031 | [-0.0014, 0.0077] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.rf.equal.size_oob` | 0.0014 | [-0.0002, 0.0029] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.rf.equal.size_oob` | -0.0004 | [-0.0008, -0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.rf.equal.size_oob` | 0.0021 | [-0.0029, 0.0068] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.rf.equal.size_uniform` | 0.0010 | [-0.0009, 0.0029] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.rf.equal.size_uniform` | -0.0002 | [-0.0006, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.rf.equal.size_uniform` | 0.0033 | [-0.0011, 0.0077] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.rf.equal.equal_oob` | 0.0008 | [-0.0008, 0.0022] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.rf.equal.equal_oob` | 0.0002 | [-0.0003, 0.0008] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.rf.equal.equal_oob` | 0.0021 | [-0.0021, 0.0058] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.rf.equal.positives_oob` | 0.0009 | [-0.0006, 0.0025] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.rf.equal.positives_oob` | 0.0003 | [-0.0004, 0.0010] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.rf.equal.positives_oob` | 0.0020 | [-0.0023, 0.0064] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.rf.equal.positives_uniform` | 0.0005 | [-0.0011, 0.0020] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.rf.equal.positives_uniform` | -0.0003 | [-0.0008, 0.0004] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.rf.equal.positives_uniform` | 0.0072 | [0.0030, 0.0115] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.rf.equal.size_oob` | 0.0013 | [-0.0003, 0.0029] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.rf.equal.size_oob` | -0.0001 | [-0.0007, 0.0005] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.rf.equal.size_oob` | 0.0014 | [-0.0029, 0.0060] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.rf.equal.size_uniform` | 0.0010 | [-0.0008, 0.0029] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.rf.equal.size_uniform` | 0.0000 | [-0.0006, 0.0007] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.rf.equal.size_uniform` | 0.0033 | [-0.0013, 0.0080] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.rf.dir0.5.equal_oob` | 0.0007 | [-0.0010, 0.0024] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.rf.dir0.5.equal_oob` | 0.0000 | [-0.0004, 0.0004] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.rf.dir0.5.equal_oob` | 0.0038 | [-0.0003, 0.0080] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.rf.dir0.5.positives_oob` | 0.0023 | [-0.0011, 0.0053] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.rf.dir0.5.positives_oob` | 0.0250 | [0.0216, 0.0287] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.rf.dir0.5.positives_oob` | -0.1106 | [-0.1241, -0.0978] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.rf.dir0.5.positives_uniform` | 0.0020 | [-0.0016, 0.0054] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.rf.dir0.5.positives_uniform` | 0.0250 | [0.0215, 0.0288] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.rf.dir0.5.positives_uniform` | -0.1107 | [-0.1220, -0.1002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.rf.dir0.5.size_oob` | 0.0126 | [0.0101, 0.0150] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.rf.dir0.5.size_oob` | -0.0090 | [-0.0099, -0.0081] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.rf.dir0.5.size_oob` | 0.0364 | [0.0280, 0.0447] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.rf.dir0.5.size_uniform` | 0.0131 | [0.0108, 0.0152] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.rf.dir0.5.size_uniform` | -0.0091 | [-0.0100, -0.0082] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.rf.dir0.5.size_uniform` | 0.0382 | [0.0301, 0.0463] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.rf.dir0.5.equal_oob` | 0.0021 | [0.0003, 0.0038] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.rf.dir0.5.equal_oob` | -0.0004 | [-0.0010, 0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.rf.dir0.5.equal_oob` | 0.0021 | [-0.0031, 0.0073] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.rf.dir0.5.positives_oob` | -0.0082 | [-0.0132, -0.0038] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.rf.dir0.5.positives_oob` | 0.0049 | [0.0031, 0.0070] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.rf.dir0.5.positives_oob` | -0.0304 | [-0.0406, -0.0208] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.rf.dir0.5.positives_uniform` | -0.0094 | [-0.0143, -0.0051] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.rf.dir0.5.positives_uniform` | 0.0050 | [0.0033, 0.0070] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.rf.dir0.5.positives_uniform` | -0.0308 | [-0.0418, -0.0207] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.rf.dir0.5.size_oob` | 0.0127 | [0.0100, 0.0154] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.rf.dir0.5.size_oob` | -0.0043 | [-0.0052, -0.0035] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.rf.dir0.5.size_oob` | 0.0174 | [0.0111, 0.0242] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.rf.dir0.5.size_uniform` | 0.0124 | [0.0097, 0.0149] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.rf.dir0.5.size_uniform` | -0.0042 | [-0.0052, -0.0033] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.rf.dir0.5.size_uniform` | 0.0130 | [0.0066, 0.0194] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.rf.dir0.5.equal_oob` | 0.0017 | [-0.0008, 0.0049] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.rf.dir0.5.equal_oob` | 0.0005 | [-0.0004, 0.0014] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.rf.dir0.5.equal_oob` | 0.0007 | [-0.0040, 0.0054] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.rf.dir0.5.positives_oob` | 0.0017 | [-0.0042, 0.0064] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.rf.dir0.5.positives_oob` | -0.0007 | [-0.0020, 0.0006] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.rf.dir0.5.positives_oob` | -0.0056 | [-0.0152, 0.0039] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.rf.dir0.5.positives_uniform` | -0.0013 | [-0.0094, 0.0052] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.rf.dir0.5.positives_uniform` | -0.0004 | [-0.0018, 0.0009] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.rf.dir0.5.positives_uniform` | -0.0072 | [-0.0184, 0.0030] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.rf.dir0.5.size_oob` | 0.0115 | [0.0072, 0.0155] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.rf.dir0.5.size_oob` | -0.0037 | [-0.0048, -0.0025] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.rf.dir0.5.size_oob` | 0.0172 | [0.0110, 0.0238] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.rf.dir0.5.size_uniform` | 0.0131 | [0.0094, 0.0169] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.rf.dir0.5.size_uniform` | -0.0039 | [-0.0049, -0.0027] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.rf.dir0.5.size_uniform` | 0.0168 | [0.0104, 0.0234] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.rf.dir0.1.equal_oob` | -0.0011 | [-0.0031, 0.0008] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.rf.dir0.1.equal_oob` | 0.0002 | [-0.0002, 0.0006] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.rf.dir0.1.equal_oob` | -0.0062 | [-0.0120, -0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.rf.dir0.1.positives_oob` | 0.0003 | [-0.0064, 0.0065] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.rf.dir0.1.positives_oob` | 0.0851 | [0.0781, 0.0924] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.rf.dir0.1.positives_oob` | -0.1329 | [-0.1459, -0.1207] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.rf.dir0.1.positives_uniform` | -0.0011 | [-0.0075, 0.0048] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.rf.dir0.1.positives_uniform` | 0.0850 | [0.0780, 0.0923] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.rf.dir0.1.positives_uniform` | -0.1325 | [-0.1454, -0.1204] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.rf.dir0.1.size_oob` | 0.0322 | [0.0281, 0.0361] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.rf.dir0.1.size_oob` | -0.0126 | [-0.0136, -0.0113] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.rf.dir0.1.size_oob` | 0.0636 | [0.0534, 0.0734] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.rf.dir0.1.size_uniform` | 0.0319 | [0.0281, 0.0358] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.rf.dir0.1.size_uniform` | -0.0125 | [-0.0135, -0.0113] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.rf.dir0.1.size_uniform` | 0.0606 | [0.0524, 0.0689] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.rf.dir0.1.equal_oob` | 0.0007 | [-0.0016, 0.0029] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.rf.dir0.1.equal_oob` | 0.0001 | [-0.0005, 0.0007] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.rf.dir0.1.equal_oob` | 0.0008 | [-0.0059, 0.0070] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.rf.dir0.1.positives_oob` | -0.0291 | [-0.0365, -0.0219] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.rf.dir0.1.positives_oob` | 0.0399 | [0.0347, 0.0453] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.rf.dir0.1.positives_oob` | -0.0619 | [-0.0745, -0.0498] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.rf.dir0.1.positives_uniform` | -0.0297 | [-0.0369, -0.0227] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.rf.dir0.1.positives_uniform` | 0.0400 | [0.0350, 0.0453] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.rf.dir0.1.positives_uniform` | -0.0652 | [-0.0791, -0.0527] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.rf.dir0.1.size_oob` | 0.0154 | [0.0110, 0.0197] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.rf.dir0.1.size_oob` | -0.0009 | [-0.0025, 0.0009] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.rf.dir0.1.size_oob` | 0.0178 | [0.0096, 0.0255] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.rf.dir0.1.size_uniform` | 0.0147 | [0.0100, 0.0193] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.rf.dir0.1.size_uniform` | -0.0009 | [-0.0026, 0.0010] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.rf.dir0.1.size_uniform` | 0.0180 | [0.0093, 0.0261] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.rf.dir0.1.equal_oob` | -0.0036 | [-0.0074, 0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.rf.dir0.1.equal_oob` | 0.0001 | [-0.0006, 0.0008] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.rf.dir0.1.equal_oob` | -0.0032 | [-0.0092, 0.0031] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.rf.dir0.1.positives_oob` | -0.0006 | [-0.0085, 0.0070] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.rf.dir0.1.positives_oob` | -0.0028 | [-0.0048, -0.0010] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.rf.dir0.1.positives_oob` | -0.0052 | [-0.0201, 0.0084] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.rf.dir0.1.positives_uniform` | -0.0040 | [-0.0124, 0.0039] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.rf.dir0.1.positives_uniform` | -0.0025 | [-0.0043, -0.0006] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.rf.dir0.1.positives_uniform` | -0.0072 | [-0.0212, 0.0057] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.rf.dir0.1.size_oob` | 0.0334 | [0.0285, 0.0381] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.rf.dir0.1.size_oob` | -0.0104 | [-0.0117, -0.0090] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.rf.dir0.1.size_oob` | 0.0539 | [0.0438, 0.0638] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.rf.dir0.1.size_uniform` | 0.0354 | [0.0300, 0.0409] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.rf.dir0.1.size_uniform` | -0.0105 | [-0.0120, -0.0090] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.rf.dir0.1.size_uniform` | 0.0557 | [0.0460, 0.0657] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.xgb.equal.positives_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.xgb.equal.positives_na` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.xgb.equal.positives_na` | -0.0003 | [-0.0007, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.xgb.equal.size_na` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.xgb.equal.size_na` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.xgb.equal.size_na` | -0.0002 | [-0.0005, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.xgb.equal.positives_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.xgb.equal.positives_na` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.xgb.equal.positives_na` | -0.0004 | [-0.0007, -0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.xgb.equal.size_na` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.xgb.equal.size_na` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.xgb.equal.size_na` | -0.0003 | [-0.0005, -0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.xgb.equal.positives_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.xgb.equal.positives_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.xgb.equal.positives_na` | -0.0001 | [-0.0004, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.xgb.equal.size_na` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.xgb.equal.size_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.xgb.equal.size_na` | 0.0001 | [-0.0002, 0.0004] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.xgb.dir0.5.positives_na` | -0.0043 | [-0.0066, -0.0019] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.xgb.dir0.5.positives_na` | 0.0670 | [0.0612, 0.0732] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.xgb.dir0.5.positives_na` | -0.1121 | [-0.1239, -0.1002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.xgb.dir0.5.size_na` | 0.0106 | [0.0090, 0.0123] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.xgb.dir0.5.size_na` | -0.0117 | [-0.0137, -0.0098] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.xgb.dir0.5.size_na` | 0.0222 | [0.0153, 0.0290] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.xgb.dir0.5.positives_na` | -0.0043 | [-0.0066, -0.0019] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.xgb.dir0.5.positives_na` | 0.0274 | [0.0246, 0.0303] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.xgb.dir0.5.positives_na` | -0.0608 | [-0.0698, -0.0522] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.xgb.dir0.5.size_na` | 0.0106 | [0.0090, 0.0123] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.xgb.dir0.5.size_na` | -0.0021 | [-0.0032, -0.0011] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.xgb.dir0.5.size_na` | 0.0119 | [0.0065, 0.0171] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.xgb.dir0.5.positives_na` | -0.0069 | [-0.0118, -0.0024] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.xgb.dir0.5.positives_na` | 0.0022 | [0.0010, 0.0033] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.xgb.dir0.5.positives_na` | -0.0127 | [-0.0211, -0.0049] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.xgb.dir0.5.size_na` | 0.0116 | [0.0092, 0.0146] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.xgb.dir0.5.size_na` | -0.0032 | [-0.0040, -0.0024] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.xgb.dir0.5.size_na` | 0.0156 | [0.0098, 0.0218] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.xgb.dir0.1.positives_na` | -0.0332 | [-0.0392, -0.0270] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.xgb.dir0.1.positives_na` | 0.1341 | [0.1268, 0.1413] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.xgb.dir0.1.positives_na` | -0.1407 | [-0.1528, -0.1288] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.diabetic.xgb.dir0.1.size_na` | 0.0071 | [0.0032, 0.0111] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.diabetic.xgb.dir0.1.size_na` | -0.0094 | [-0.0120, -0.0067] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.diabetic.xgb.dir0.1.size_na` | 0.0189 | [0.0086, 0.0287] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.xgb.dir0.1.positives_na` | -0.0332 | [-0.0392, -0.0270] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.xgb.dir0.1.positives_na` | 0.0649 | [0.0611, 0.0686] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.xgb.dir0.1.positives_na` | -0.0804 | [-0.0914, -0.0704] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.diabetic.xgb.dir0.1.size_na` | 0.0071 | [0.0032, 0.0111] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.diabetic.xgb.dir0.1.size_na` | 0.0048 | [0.0031, 0.0065] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.diabetic.xgb.dir0.1.size_na` | 0.0098 | [0.0022, 0.0177] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.xgb.dir0.1.positives_na` | -0.0408 | [-0.0500, -0.0317] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.xgb.dir0.1.positives_na` | 0.0062 | [0.0042, 0.0080] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.xgb.dir0.1.positives_na` | -0.0697 | [-0.0821, -0.0569] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.diabetic.xgb.dir0.1.size_na` | 0.0066 | [0.0023, 0.0107] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.diabetic.xgb.dir0.1.size_na` | -0.0028 | [-0.0042, -0.0014] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.diabetic.xgb.dir0.1.size_na` | 0.0073 | [-0.0007, 0.0149] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.rf.equal.equal_oob` | 0.0001 | [-0.0002, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.rf.equal.equal_oob` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.rf.equal.equal_oob` | 0.0002 | [-0.0001, 0.0004] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.rf.equal.positives_oob` | 0.0001 | [-0.0001, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.rf.equal.positives_oob` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.rf.equal.positives_oob` | -0.0000 | [-0.0003, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.rf.equal.positives_uniform` | -0.0000 | [-0.0002, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.rf.equal.positives_uniform` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.rf.equal.positives_uniform` | -0.0000 | [-0.0003, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.rf.equal.size_oob` | 0.0001 | [-0.0001, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.rf.equal.size_oob` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.rf.equal.size_oob` | 0.0000 | [-0.0002, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.rf.equal.size_uniform` | 0.0001 | [-0.0000, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.rf.equal.size_uniform` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.rf.equal.size_uniform` | 0.0002 | [-0.0001, 0.0004] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.rf.equal.equal_oob` | 0.0001 | [-0.0002, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.rf.equal.equal_oob` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.rf.equal.equal_oob` | 0.0002 | [-0.0001, 0.0004] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.rf.equal.positives_oob` | 0.0001 | [-0.0002, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.rf.equal.positives_oob` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.rf.equal.positives_oob` | -0.0000 | [-0.0003, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.rf.equal.positives_uniform` | -0.0000 | [-0.0002, 0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.rf.equal.positives_uniform` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.rf.equal.positives_uniform` | -0.0000 | [-0.0003, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.rf.equal.size_oob` | 0.0001 | [-0.0001, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.rf.equal.size_oob` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.rf.equal.size_oob` | 0.0000 | [-0.0002, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.rf.equal.size_uniform` | 0.0001 | [-0.0000, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.rf.equal.size_uniform` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.rf.equal.size_uniform` | 0.0002 | [-0.0001, 0.0004] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.rf.equal.equal_oob` | 0.0001 | [-0.0002, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.rf.equal.equal_oob` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.rf.equal.equal_oob` | 0.0001 | [-0.0002, 0.0004] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.rf.equal.positives_oob` | 0.0001 | [-0.0001, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.rf.equal.positives_oob` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.rf.equal.positives_oob` | -0.0001 | [-0.0004, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.rf.equal.positives_uniform` | -0.0000 | [-0.0002, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.rf.equal.positives_uniform` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.rf.equal.positives_uniform` | -0.0000 | [-0.0003, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.rf.equal.size_oob` | 0.0001 | [-0.0001, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.rf.equal.size_oob` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.rf.equal.size_oob` | 0.0001 | [-0.0003, 0.0004] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.rf.equal.size_uniform` | 0.0001 | [-0.0000, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.rf.equal.size_uniform` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.rf.equal.size_uniform` | -0.0000 | [-0.0004, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.rf.dir0.5.equal_oob` | 0.0000 | [-0.0001, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.rf.dir0.5.equal_oob` | -0.0000 | [-0.0001, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.rf.dir0.5.equal_oob` | -0.0002 | [-0.0007, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.rf.dir0.5.positives_oob` | 0.0036 | [0.0030, 0.0041] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.rf.dir0.5.positives_oob` | 0.0416 | [0.0342, 0.0490] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.rf.dir0.5.positives_oob` | -0.0133 | [-0.0170, -0.0093] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.rf.dir0.5.positives_uniform` | 0.0036 | [0.0030, 0.0041] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.rf.dir0.5.positives_uniform` | 0.0416 | [0.0343, 0.0490] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.rf.dir0.5.positives_uniform` | -0.0132 | [-0.0169, -0.0092] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.rf.dir0.5.size_oob` | -0.0096 | [-0.0112, -0.0083] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.rf.dir0.5.size_oob` | -0.0175 | [-0.0204, -0.0150] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.rf.dir0.5.size_oob` | 0.1053 | [0.0986, 0.1111] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.rf.dir0.5.size_uniform` | -0.0096 | [-0.0112, -0.0082] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.rf.dir0.5.size_uniform` | -0.0175 | [-0.0204, -0.0150] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.rf.dir0.5.size_uniform` | 0.1053 | [0.0987, 0.1113] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.rf.dir0.5.equal_oob` | -0.0000 | [-0.0003, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.rf.dir0.5.equal_oob` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.rf.dir0.5.equal_oob` | -0.0002 | [-0.0007, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.rf.dir0.5.positives_oob` | 0.0117 | [0.0108, 0.0126] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.rf.dir0.5.positives_oob` | 0.0046 | [0.0019, 0.0081] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.rf.dir0.5.positives_oob` | 0.0092 | [0.0080, 0.0106] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.rf.dir0.5.positives_uniform` | 0.0116 | [0.0107, 0.0126] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.rf.dir0.5.positives_uniform` | 0.0046 | [0.0019, 0.0081] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.rf.dir0.5.positives_uniform` | 0.0091 | [0.0078, 0.0104] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.rf.dir0.5.size_oob` | -0.0067 | [-0.0083, -0.0051] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.rf.dir0.5.size_oob` | -0.0015 | [-0.0023, -0.0008] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.rf.dir0.5.size_oob` | 0.0027 | [-0.0004, 0.0059] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.rf.dir0.5.size_uniform` | -0.0066 | [-0.0082, -0.0049] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.rf.dir0.5.size_uniform` | -0.0015 | [-0.0023, -0.0008] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.rf.dir0.5.size_uniform` | 0.0031 | [0.0000, 0.0063] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.rf.dir0.5.equal_oob` | 0.0000 | [-0.0001, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.rf.dir0.5.equal_oob` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.rf.dir0.5.equal_oob` | -0.0002 | [-0.0006, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.rf.dir0.5.positives_oob` | 0.0036 | [0.0030, 0.0041] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.rf.dir0.5.positives_oob` | -0.0002 | [-0.0002, -0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.rf.dir0.5.positives_oob` | 0.0038 | [0.0033, 0.0044] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.rf.dir0.5.positives_uniform` | 0.0036 | [0.0030, 0.0041] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.rf.dir0.5.positives_uniform` | -0.0002 | [-0.0002, -0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.rf.dir0.5.positives_uniform` | 0.0040 | [0.0034, 0.0045] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.rf.dir0.5.size_oob` | -0.0096 | [-0.0112, -0.0083] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.rf.dir0.5.size_oob` | 0.0002 | [0.0002, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.rf.dir0.5.size_oob` | -0.0080 | [-0.0090, -0.0070] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.rf.dir0.5.size_uniform` | -0.0096 | [-0.0112, -0.0082] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.rf.dir0.5.size_uniform` | 0.0002 | [0.0002, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.rf.dir0.5.size_uniform` | -0.0077 | [-0.0089, -0.0066] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.rf.dir0.1.equal_oob` | -0.0007 | [-0.0020, 0.0005] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.rf.dir0.1.equal_oob` | -0.0002 | [-0.0008, 0.0004] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.rf.dir0.1.equal_oob` | 0.0009 | [0.0004, 0.0014] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.rf.dir0.1.positives_oob` | -0.0118 | [-0.0182, -0.0054] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.rf.dir0.1.positives_oob` | 0.2859 | [0.2567, 0.3151] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.rf.dir0.1.positives_oob` | 0.0234 | [0.0176, 0.0297] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.rf.dir0.1.positives_uniform` | -0.0123 | [-0.0187, -0.0057] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.rf.dir0.1.positives_uniform` | 0.2860 | [0.2569, 0.3152] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.rf.dir0.1.positives_uniform` | 0.0233 | [0.0175, 0.0296] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.rf.dir0.1.size_oob` | -0.0398 | [-0.0472, -0.0327] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.rf.dir0.1.size_oob` | -0.0957 | [-0.1031, -0.0890] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.rf.dir0.1.size_oob` | 0.1695 | [0.1629, 0.1764] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.rf.dir0.1.size_uniform` | -0.0400 | [-0.0472, -0.0330] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.rf.dir0.1.size_uniform` | -0.0957 | [-0.1031, -0.0890] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.rf.dir0.1.size_uniform` | 0.1696 | [0.1628, 0.1763] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.rf.dir0.1.equal_oob` | -0.0000 | [-0.0010, 0.0010] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.rf.dir0.1.equal_oob` | 0.0002 | [-0.0000, 0.0005] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.rf.dir0.1.equal_oob` | 0.0006 | [-0.0003, 0.0015] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.rf.dir0.1.positives_oob` | -0.0058 | [-0.0119, 0.0006] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.rf.dir0.1.positives_oob` | 0.1893 | [0.1630, 0.2137] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.rf.dir0.1.positives_oob` | 0.0231 | [0.0206, 0.0256] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.rf.dir0.1.positives_uniform` | -0.0063 | [-0.0125, 0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.rf.dir0.1.positives_uniform` | 0.1894 | [0.1631, 0.2141] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.rf.dir0.1.positives_uniform` | 0.0231 | [0.0207, 0.0255] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.rf.dir0.1.size_oob` | -0.0284 | [-0.0361, -0.0210] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.rf.dir0.1.size_oob` | -0.0301 | [-0.0337, -0.0267] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.rf.dir0.1.size_oob` | 0.0641 | [0.0581, 0.0698] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.rf.dir0.1.size_uniform` | -0.0286 | [-0.0361, -0.0213] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.rf.dir0.1.size_uniform` | -0.0301 | [-0.0337, -0.0267] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.rf.dir0.1.size_uniform` | 0.0637 | [0.0576, 0.0695] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.rf.dir0.1.equal_oob` | -0.0007 | [-0.0020, 0.0005] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.rf.dir0.1.equal_oob` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.rf.dir0.1.equal_oob` | -0.0013 | [-0.0025, -0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.rf.dir0.1.positives_oob` | -0.0118 | [-0.0182, -0.0054] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.rf.dir0.1.positives_oob` | 0.0000 | [-0.0001, 0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.rf.dir0.1.positives_oob` | -0.0014 | [-0.0057, 0.0032] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.rf.dir0.1.positives_uniform` | -0.0123 | [-0.0187, -0.0057] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.rf.dir0.1.positives_uniform` | 0.0000 | [-0.0001, 0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.rf.dir0.1.positives_uniform` | -0.0014 | [-0.0057, 0.0030] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.rf.dir0.1.size_oob` | -0.0398 | [-0.0472, -0.0327] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.rf.dir0.1.size_oob` | 0.0004 | [0.0002, 0.0005] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.rf.dir0.1.size_oob` | -0.0180 | [-0.0223, -0.0138] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.rf.dir0.1.size_uniform` | -0.0400 | [-0.0472, -0.0330] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.rf.dir0.1.size_uniform` | 0.0004 | [0.0002, 0.0005] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.rf.dir0.1.size_uniform` | -0.0178 | [-0.0219, -0.0139] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.xgb.equal.positives_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.xgb.equal.positives_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.xgb.equal.positives_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.xgb.equal.size_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.xgb.equal.size_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.xgb.equal.size_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.xgb.equal.positives_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.xgb.equal.positives_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.xgb.equal.positives_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.xgb.equal.size_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.xgb.equal.size_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.xgb.equal.size_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.xgb.equal.positives_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.xgb.equal.positives_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.xgb.equal.positives_na` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.xgb.equal.size_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.xgb.equal.size_na` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.xgb.equal.size_na` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.xgb.dir0.5.positives_na` | 0.0036 | [0.0028, 0.0044] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.xgb.dir0.5.positives_na` | 0.0581 | [0.0472, 0.0689] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.xgb.dir0.5.positives_na` | -0.0302 | [-0.0368, -0.0234] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.xgb.dir0.5.size_na` | -0.0142 | [-0.0160, -0.0124] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.xgb.dir0.5.size_na` | -0.0044 | [-0.0072, -0.0019] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.xgb.dir0.5.size_na` | -0.0571 | [-0.0649, -0.0491] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.xgb.dir0.5.positives_na` | 0.0036 | [0.0028, 0.0044] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.xgb.dir0.5.positives_na` | 0.0034 | [0.0018, 0.0052] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.xgb.dir0.5.positives_na` | 0.0015 | [-0.0007, 0.0038] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.xgb.dir0.5.size_na` | -0.0142 | [-0.0160, -0.0124] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.xgb.dir0.5.size_na` | 0.0004 | [0.0001, 0.0007] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.xgb.dir0.5.size_na` | -0.0241 | [-0.0276, -0.0208] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.xgb.dir0.5.positives_na` | 0.0036 | [0.0028, 0.0044] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.xgb.dir0.5.positives_na` | -0.0002 | [-0.0002, -0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.xgb.dir0.5.positives_na` | 0.0031 | [0.0023, 0.0039] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.xgb.dir0.5.size_na` | -0.0142 | [-0.0160, -0.0124] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.xgb.dir0.5.size_na` | 0.0006 | [0.0006, 0.0007] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.xgb.dir0.5.size_na` | -0.0145 | [-0.0162, -0.0127] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.xgb.dir0.1.positives_na` | -0.0166 | [-0.0207, -0.0123] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.xgb.dir0.1.positives_na` | 0.4115 | [0.3771, 0.4450] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.xgb.dir0.1.positives_na` | -0.0784 | [-0.0886, -0.0685] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L0.heart.xgb.dir0.1.size_na` | -0.0458 | [-0.0503, -0.0411] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L0.heart.xgb.dir0.1.size_na` | -0.0513 | [-0.0644, -0.0399] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L0.heart.xgb.dir0.1.size_na` | -0.0828 | [-0.0952, -0.0703] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.xgb.dir0.1.positives_na` | -0.0166 | [-0.0207, -0.0123] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.xgb.dir0.1.positives_na` | 0.0701 | [0.0621, 0.0787] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.xgb.dir0.1.positives_na` | -0.0361 | [-0.0423, -0.0301] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L1.heart.xgb.dir0.1.size_na` | -0.0458 | [-0.0503, -0.0411] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L1.heart.xgb.dir0.1.size_na` | 0.0005 | [-0.0000, 0.0010] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L1.heart.xgb.dir0.1.size_na` | -0.0572 | [-0.0631, -0.0510] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.xgb.dir0.1.positives_na` | -0.0166 | [-0.0207, -0.0123] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.xgb.dir0.1.positives_na` | 0.0002 | [0.0001, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.xgb.dir0.1.positives_na` | -0.0149 | [-0.0186, -0.0111] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.auroc.fed_L2.heart.xgb.dir0.1.size_na` | -0.0458 | [-0.0503, -0.0411] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.brier.fed_L2.heart.xgb.dir0.1.size_na` | 0.0010 | [0.0009, 0.0012] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `e5.mcc_pi.fed_L2.heart.xgb.dir0.1.size_na` | -0.0391 | [-0.0430, -0.0352] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Configuration minus the default (equal weights, uniform sampling). |
| `rfxgb.auroc.L3+L1.aids.equal` | 0.0032 | [-0.0004, 0.0070] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L2.aids.equal` | 0.0021 | [-0.0025, 0.0065] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L1.aids.equal` | -0.0180 | [-0.0193, -0.0168] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L2.aids.equal` | 0.0003 | [-0.0004, 0.0011] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L1.aids.dir0.5` | -0.0080 | [-0.0111, -0.0046] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L2.aids.dir0.5` | -0.0011 | [-0.0051, 0.0032] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L1.aids.dir0.5` | -0.0097 | [-0.0108, -0.0087] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L2.aids.dir0.5` | -0.0000 | [-0.0006, 0.0005] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L1.aids.dir0.1` | -0.0159 | [-0.0198, -0.0122] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L2.aids.dir0.1` | -0.0116 | [-0.0185, -0.0045] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L1.aids.dir0.1` | 0.0044 | [0.0029, 0.0059] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L2.aids.dir0.1` | 0.0005 | [-0.0000, 0.0010] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L1.diabetic.equal` | -0.0229 | [-0.0259, -0.0199] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L2.diabetic.equal` | -0.0241 | [-0.0277, -0.0208] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L1.diabetic.equal` | -0.0025 | [-0.0051, 0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L2.diabetic.equal` | 0.0087 | [0.0071, 0.0104] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L1.diabetic.dir0.5` | -0.0235 | [-0.0272, -0.0199] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L2.diabetic.dir0.5` | -0.0157 | [-0.0191, -0.0121] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L1.diabetic.dir0.5` | -0.0008 | [-0.0029, 0.0013] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L2.diabetic.dir0.5` | 0.0057 | [0.0041, 0.0073] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L1.diabetic.dir0.1` | -0.0224 | [-0.0265, -0.0180] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L2.diabetic.dir0.1` | -0.0225 | [-0.0281, -0.0168] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L1.diabetic.dir0.1` | 0.0024 | [0.0005, 0.0040] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L2.diabetic.dir0.1` | 0.0068 | [0.0052, 0.0084] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L1.heart.equal` | -0.0129 | [-0.0135, -0.0122] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L2.heart.equal` | -0.0129 | [-0.0135, -0.0122] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L1.heart.equal` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L2.heart.equal` | 0.0004 | [0.0003, 0.0004] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L1.heart.dir0.5` | -0.0178 | [-0.0188, -0.0169] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L2.heart.dir0.5` | -0.0026 | [-0.0033, -0.0018] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L1.heart.dir0.5` | 0.0013 | [0.0007, 0.0020] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L2.heart.dir0.5` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L1.heart.dir0.1` | -0.0353 | [-0.0385, -0.0324] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.auroc.L3+L2.heart.dir0.1` | -0.0198 | [-0.0231, -0.0167] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L1.heart.dir0.1` | 0.0309 | [0.0276, 0.0345] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `rfxgb.brier.L3+L2.heart.dir0.1` | 0.0003 | [0.0002, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. rf minus xgb. |
| `l1g.auroc.aids.rf.equal` | -0.0000 | [-0.0001, 0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.aids.rf.equal` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.aids.rf.equal` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.aids.rf.dir0.5` | 0.0089 | [0.0074, 0.0104] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.aids.rf.dir0.5` | -0.0028 | [-0.0035, -0.0021] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.aids.rf.dir0.5` | 0.0185 | [0.0148, 0.0221] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.aids.rf.dir0.1` | 0.0092 | [0.0069, 0.0113] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.aids.rf.dir0.1` | -0.0114 | [-0.0127, -0.0100] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.aids.rf.dir0.1` | 0.0363 | [0.0306, 0.0423] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.aids.xgb.equal` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.aids.xgb.equal` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.aids.xgb.equal` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.aids.xgb.dir0.5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.aids.xgb.dir0.5` | 0.0042 | [0.0037, 0.0048] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.aids.xgb.dir0.5` | -0.0086 | [-0.0129, -0.0042] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.aids.xgb.dir0.1` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.aids.xgb.dir0.1` | 0.0036 | [0.0029, 0.0044] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.aids.xgb.dir0.1` | -0.0070 | [-0.0109, -0.0034] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.diabetic.rf.equal` | -0.0001 | [-0.0002, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.diabetic.rf.equal` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.diabetic.rf.equal` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.diabetic.rf.dir0.5` | 0.0078 | [0.0062, 0.0094] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.diabetic.rf.dir0.5` | -0.0003 | [-0.0010, 0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.diabetic.rf.dir0.5` | 0.0114 | [0.0064, 0.0169] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.diabetic.rf.dir0.1` | -0.0005 | [-0.0030, 0.0018] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.diabetic.rf.dir0.1` | -0.0008 | [-0.0017, 0.0001] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.diabetic.rf.dir0.1` | -0.0028 | [-0.0085, 0.0033] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.diabetic.xgb.equal` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.diabetic.xgb.equal` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.diabetic.xgb.equal` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.diabetic.xgb.dir0.5` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.diabetic.xgb.dir0.5` | 0.0025 | [0.0020, 0.0031] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.diabetic.xgb.dir0.5` | -0.0055 | [-0.0083, -0.0030] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.diabetic.xgb.dir0.1` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.diabetic.xgb.dir0.1` | 0.0047 | [0.0040, 0.0054] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.diabetic.xgb.dir0.1` | -0.0068 | [-0.0106, -0.0030] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.heart.rf.equal` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.heart.rf.equal` | 0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.heart.rf.equal` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.heart.rf.dir0.5` | 0.0152 | [0.0146, 0.0159] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.heart.rf.dir0.5` | -0.0020 | [-0.0027, -0.0013] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.heart.rf.dir0.5` | 0.0237 | [0.0205, 0.0269] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.heart.rf.dir0.1` | 0.0155 | [0.0138, 0.0173] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.heart.rf.dir0.1` | -0.0317 | [-0.0355, -0.0283] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.heart.rf.dir0.1` | 0.0943 | [0.0886, 0.1000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.heart.xgb.equal` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.heart.xgb.equal` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.heart.xgb.equal` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.heart.xgb.dir0.5` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.heart.xgb.dir0.5` | 0.0022 | [0.0019, 0.0025] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.heart.xgb.dir0.5` | -0.0446 | [-0.0490, -0.0402] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.auroc.heart.xgb.dir0.1` | -0.0000 | [-0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.brier.heart.xgb.dir0.1` | 0.0046 | [0.0034, 0.0059] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `l1g.mcc_pi.heart.xgb.dir0.1` | -0.0698 | [-0.0758, -0.0634] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L1g minus fed_L1. |
| `e10.r.brier.L1.shift_double.aids.rf.equal` | 0.0003 | [0.0001, 0.0004] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.aids.rf.equal` | 0.6416 | [0.6023, 0.6811] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.aids.rf.equal` | 0.0002 | [0.0001, 0.0003] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.aids.rf.equal` | 0.8745 | [0.8418, 0.9046] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.aids.rf.dir0.5` | 0.7054 | [0.6936, 0.7169] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.aids.rf.dir0.5` | 0.9021 | [0.8909, 0.9131] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.aids.rf.dir0.5` | 0.8451 | [0.8344, 0.8548] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.aids.rf.dir0.5` | 0.9797 | [0.9748, 0.9856] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.aids.rf.dir0.1` | 0.5696 | [0.5593, 0.5798] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.aids.rf.dir0.1` | 0.9477 | [0.9399, 0.9559] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.aids.rf.dir0.1` | 0.8039 | [0.7948, 0.8130] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.aids.rf.dir0.1` | 0.9911 | [0.9891, 0.9930] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.aids.xgb.equal` | 0.0000 | [-0.0000, 0.0001] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.aids.xgb.equal` | 0.8368 | [0.8175, 0.8544] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.aids.xgb.equal` | 0.0000 | [-0.0000, 0.0001] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.aids.xgb.equal` | 0.9266 | [0.9067, 0.9485] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.aids.xgb.dir0.5` | 0.4910 | [0.4813, 0.5005] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.aids.xgb.dir0.5` | 0.9141 | [0.9060, 0.9216] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.aids.xgb.dir0.5` | 0.4743 | [0.4644, 0.4836] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.aids.xgb.dir0.5` | 0.9805 | [0.9763, 0.9852] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.aids.xgb.dir0.1` | 0.5685 | [0.5604, 0.5767] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.aids.xgb.dir0.1` | 0.9498 | [0.9432, 0.9566] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.aids.xgb.dir0.1` | 0.6024 | [0.5954, 0.6100] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.aids.xgb.dir0.1` | 0.9901 | [0.9882, 0.9920] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.diabetic.rf.equal` | 0.0002 | [0.0001, 0.0003] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.diabetic.rf.equal` | 0.6885 | [0.6422, 0.7341] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.diabetic.rf.equal` | 0.0002 | [0.0001, 0.0003] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.diabetic.rf.equal` | 0.5536 | [0.5033, 0.6097] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.diabetic.rf.dir0.5` | 0.6237 | [0.6075, 0.6401] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.diabetic.rf.dir0.5` | 0.8941 | [0.8813, 0.9071] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.diabetic.rf.dir0.5` | 0.6206 | [0.6072, 0.6334] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.diabetic.rf.dir0.5` | 0.8483 | [0.8307, 0.8688] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.diabetic.rf.dir0.1` | 0.5345 | [0.5201, 0.5488] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.diabetic.rf.dir0.1` | 0.9572 | [0.9474, 0.9685] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.diabetic.rf.dir0.1` | 0.5115 | [0.4984, 0.5247] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.diabetic.rf.dir0.1` | 0.9295 | [0.9164, 0.9449] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.diabetic.xgb.equal` | 0.0000 | [-0.0000, 0.0000] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.diabetic.xgb.equal` | 0.8157 | [0.7861, 0.8487] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.diabetic.xgb.equal` | -0.0000 | [-0.0000, 0.0000] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.diabetic.xgb.equal` | 0.7249 | [0.7012, 0.7518] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.diabetic.xgb.dir0.5` | 0.3592 | [0.3474, 0.3704] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.diabetic.xgb.dir0.5` | 0.8965 | [0.8840, 0.9088] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.diabetic.xgb.dir0.5` | 0.3285 | [0.3179, 0.3387] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.diabetic.xgb.dir0.5` | 0.8467 | [0.8322, 0.8621] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.diabetic.xgb.dir0.1` | 0.4271 | [0.4138, 0.4399] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.diabetic.xgb.dir0.1` | 0.9299 | [0.9199, 0.9403] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.diabetic.xgb.dir0.1` | 0.3928 | [0.3827, 0.4034] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.diabetic.xgb.dir0.1` | 0.8922 | [0.8782, 0.9065] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.heart.rf.equal` | 0.0000 | [-0.0000, 0.0000] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.heart.rf.equal` | 0.7275 | [0.7237, 0.7313] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.heart.rf.equal` | 0.0000 | [-0.0000, 0.0000] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.heart.rf.equal` | 0.9529 | [0.9513, 0.9545] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.heart.rf.dir0.5` | 0.8730 | [0.8519, 0.8927] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.heart.rf.dir0.5` | 0.9386 | [0.9323, 0.9439] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.heart.rf.dir0.5` | 0.9259 | [0.9065, 0.9432] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.heart.rf.dir0.5` | 0.9939 | [0.9933, 0.9945] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.heart.rf.dir0.1` | 0.6571 | [0.6372, 0.6774] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.heart.rf.dir0.1` | 0.9761 | [0.9746, 0.9775] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.heart.rf.dir0.1` | 0.7258 | [0.7050, 0.7465] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.heart.rf.dir0.1` | 0.9979 | [0.9978, 0.9981] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.heart.xgb.equal` | -0.0000 | [-0.0000, 0.0000] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.heart.xgb.equal` | 0.5349 | [0.5316, 0.5385] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.heart.xgb.equal` | -0.0000 | [-0.0000, 0.0000] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.heart.xgb.equal` | 0.8724 | [0.8693, 0.8753] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.heart.xgb.dir0.5` | 0.8408 | [0.8311, 0.8509] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.heart.xgb.dir0.5` | 0.9320 | [0.9257, 0.9374] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.heart.xgb.dir0.5` | 0.7967 | [0.7889, 0.8048] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.heart.xgb.dir0.5` | 0.9926 | [0.9919, 0.9932] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_double.heart.xgb.dir0.1` | 0.8391 | [0.8328, 0.8459] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_double.heart.xgb.dir0.1` | 0.9743 | [0.9729, 0.9756] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L1.shift_half.heart.xgb.dir0.1` | 0.8200 | [0.8138, 0.8266] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e10.r.brier.L2.shift_half.heart.xgb.dir0.1` | 0.9977 | [0.9976, 0.9979] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Shifted calibration and test parts. Reportable: yes. |
| `e11.r.brier.L1.aids.rf.equal` | 0.0077 | [0.0070, 0.0084] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.aids.rf.equal` | 0.1745 | [0.0200, 0.2989] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.aids.rf.dir0.5` | 0.4231 | [0.3803, 0.4687] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.aids.rf.dir0.5` | 0.8465 | [0.8234, 0.8679] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.aids.rf.dir0.1` | 0.2420 | [0.1777, 0.3066] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.aids.rf.dir0.1` | 0.9310 | [0.9137, 0.9485] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.aids.xgb.equal` | 0.0023 | [0.0021, 0.0024] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.aids.xgb.equal` | 0.8312 | [0.8072, 0.8525] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.aids.xgb.dir0.5` | 0.2360 | [0.2141, 0.2581] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.aids.xgb.dir0.5` | 0.8760 | [0.8622, 0.8892] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.aids.xgb.dir0.1` | 0.3064 | [0.2895, 0.3244] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.aids.xgb.dir0.1` | 0.9299 | [0.9185, 0.9416] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.diabetic.rf.equal` | 0.0004 | [-0.0001, 0.0009] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.diabetic.rf.equal` | -0.1868 | [-0.4400, -0.0549] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.diabetic.rf.dir0.5` | 0.7777 | [0.7441, 0.8126] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.diabetic.rf.dir0.5` | 0.8950 | [0.8680, 0.9211] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.diabetic.rf.dir0.1` | 0.6067 | [0.5708, 0.6417] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.diabetic.rf.dir0.1` | 1.0115 | [0.9943, 1.0317] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.diabetic.xgb.equal` | 0.0001 | [0.0000, 0.0003] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.diabetic.xgb.equal` | 0.6618 | [0.6335, 0.6924] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.diabetic.xgb.dir0.5` | 0.3828 | [0.3647, 0.4019] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.diabetic.xgb.dir0.5` | 0.9036 | [0.8849, 0.9226] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.diabetic.xgb.dir0.1` | 0.4858 | [0.4714, 0.4997] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.diabetic.xgb.dir0.1` | 0.9696 | [0.9574, 0.9823] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.heart.rf.equal` | 0.0011 | [0.0011, 0.0012] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.heart.rf.equal` | 0.3002 | [0.2809, 0.3182] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.heart.rf.dir0.5` | -1.5999 | [-13.6245, 10.6318] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: no. |
| `e11.r.brier.L2.heart.rf.dir0.5` | 0.5451 | [0.5262, 0.5641] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.heart.rf.dir0.1` | 0.0802 | [0.0662, 0.0965] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.heart.rf.dir0.1` | 0.5925 | [0.5692, 0.6192] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.heart.xgb.equal` | 0.0002 | [0.0002, 0.0002] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.heart.xgb.equal` | 0.5127 | [0.5053, 0.5200] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.heart.xgb.dir0.5` | 0.3810 | [0.3246, 0.4564] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.heart.xgb.dir0.5` | 0.6145 | [0.6010, 0.6283] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L1.heart.xgb.dir0.1` | 0.3161 | [0.2190, 0.4960] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.r.brier.L2.heart.xgb.dir0.1` | 0.6895 | [0.6700, 0.7112] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Largest site as the source. Reportable: yes. |
| `e11.diff.auroc.L3.aids.rf.equal` | 0.0336 | [0.0305, 0.0368] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.aids.rf.dir0.5` | 0.0351 | [0.0299, 0.0403] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.aids.rf.dir0.1` | 0.0294 | [0.0245, 0.0349] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.aids.xgb.equal` | 0.0594 | [0.0556, 0.0632] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.aids.xgb.dir0.5` | 0.0526 | [0.0479, 0.0572] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.aids.xgb.dir0.1` | 0.0405 | [0.0357, 0.0454] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.diabetic.rf.equal` | 0.0406 | [0.0365, 0.0448] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.diabetic.rf.dir0.5` | 0.0212 | [0.0167, 0.0259] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.diabetic.rf.dir0.1` | 0.0142 | [0.0063, 0.0226] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.diabetic.xgb.equal` | 0.0661 | [0.0610, 0.0709] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.diabetic.xgb.dir0.5` | 0.0318 | [0.0270, 0.0368] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.diabetic.xgb.dir0.1` | 0.0329 | [0.0258, 0.0403] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.heart.rf.equal` | 0.0075 | [0.0073, 0.0077] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.heart.rf.dir0.5` | 0.0925 | [0.0833, 0.1023] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.heart.rf.dir0.1` | 0.1530 | [0.1410, 0.1648] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.heart.xgb.equal` | 0.0295 | [0.0292, 0.0298] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.heart.xgb.dir0.5` | 0.0531 | [0.0485, 0.0581] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `e11.diff.auroc.L3.heart.xgb.dir0.1` | 0.0759 | [0.0691, 0.0828] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L0 minus the largest source. |
| `l2neg.fed_L2.aids.rf.equal.n25.share` | 0.1089 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.rf.equal.n25.dauroc_vs_src_L0` | -0.0088 | [-0.0345, 0.0141] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.rf.equal.n25.dauroc_vs_fed_L0` | -0.0428 | [-0.0695, -0.0193] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.aids.rf.equal.n50.share` | 0.0622 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.rf.equal.n50.dauroc_vs_src_L0` | 0.0085 | [-0.0188, 0.0296] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.rf.equal.n50.dauroc_vs_fed_L0` | -0.0254 | [-0.0533, -0.0037] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.aids.rf.equal.nfull.share` | 0.0022 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.rf.equal.nfull.dauroc_vs_src_L0` | 0.0329 | [0.0302, 0.0354] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.rf.equal.nfull.dauroc_vs_fed_L0` | -0.0010 | [-0.0031, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.aids.rf.equal.n25.share` | 0.1674 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.rf.equal.n25.dauroc_vs_src_L0` | -0.0471 | [-0.0640, -0.0317] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.aids.rf.equal.n50.share` | 0.0885 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.rf.equal.n50.dauroc_vs_src_L0` | -0.0254 | [-0.0417, -0.0129] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.aids.rf.equal.nfull.share` | 0.0389 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.rf.equal.nfull.dauroc_vs_src_L0` | -0.0101 | [-0.0160, -0.0052] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.rf.dir0.5.n25.share` | 0.1378 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.rf.dir0.5.n25.dauroc_vs_src_L0` | 0.0033 | [-0.0167, 0.0214] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.rf.dir0.5.n25.dauroc_vs_fed_L0` | -0.0425 | [-0.0629, -0.0249] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.aids.rf.dir0.5.n50.share` | 0.0578 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.rf.dir0.5.n50.dauroc_vs_src_L0` | 0.0264 | [0.0101, 0.0389] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.rf.dir0.5.n50.dauroc_vs_fed_L0` | -0.0194 | [-0.0363, -0.0067] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.aids.rf.dir0.5.nfull.share` | 0.0111 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.rf.dir0.5.nfull.dauroc_vs_src_L0` | 0.0426 | [0.0389, 0.0462] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.rf.dir0.5.nfull.dauroc_vs_fed_L0` | -0.0032 | [-0.0062, -0.0008] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.aids.rf.dir0.5.n25.share` | 0.2396 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.rf.dir0.5.n25.dauroc_vs_src_L0` | -0.0433 | [-0.0556, -0.0323] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.aids.rf.dir0.5.n50.share` | 0.1615 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.rf.dir0.5.n50.dauroc_vs_src_L0` | -0.0265 | [-0.0376, -0.0178] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.aids.rf.dir0.5.nfull.share` | 0.0930 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.rf.dir0.5.nfull.dauroc_vs_src_L0` | -0.0114 | [-0.0156, -0.0078] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.rf.dir0.1.n25.share` | 0.2244 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.rf.dir0.1.n25.dauroc_vs_src_L0` | 0.0005 | [-0.0143, 0.0135] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.rf.dir0.1.n25.dauroc_vs_fed_L0` | -0.0469 | [-0.0615, -0.0338] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.aids.rf.dir0.1.n50.share` | 0.1400 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.rf.dir0.1.n50.dauroc_vs_src_L0` | 0.0218 | [0.0108, 0.0312] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.rf.dir0.1.n50.dauroc_vs_fed_L0` | -0.0256 | [-0.0370, -0.0157] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.aids.rf.dir0.1.nfull.share` | 0.1022 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.rf.dir0.1.nfull.dauroc_vs_src_L0` | 0.0315 | [0.0248, 0.0373] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.rf.dir0.1.nfull.dauroc_vs_fed_L0` | -0.0160 | [-0.0232, -0.0103] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.aids.rf.dir0.1.n25.share` | 0.3219 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.rf.dir0.1.n25.dauroc_vs_src_L0` | -0.0357 | [-0.0440, -0.0279] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.aids.rf.dir0.1.n50.share` | 0.2611 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.rf.dir0.1.n50.dauroc_vs_src_L0` | -0.0217 | [-0.0292, -0.0152] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.aids.rf.dir0.1.nfull.share` | 0.2133 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.rf.dir0.1.nfull.dauroc_vs_src_L0` | -0.0118 | [-0.0158, -0.0080] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.xgb.equal.n25.share` | 0.0889 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.xgb.equal.n25.dauroc_vs_src_L0` | 0.0269 | [0.0073, 0.0425] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.xgb.equal.n25.dauroc_vs_fed_L0` | -0.0319 | [-0.0523, -0.0155] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.aids.xgb.equal.n50.share` | 0.0378 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.xgb.equal.n50.dauroc_vs_src_L0` | 0.0431 | [0.0252, 0.0573] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.xgb.equal.n50.dauroc_vs_fed_L0` | -0.0158 | [-0.0343, -0.0004] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.aids.xgb.equal.nfull.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.xgb.equal.nfull.dauroc_vs_src_L0` | 0.0588 | [0.0559, 0.0620] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.xgb.equal.nfull.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.aids.xgb.equal.n25.share` | 0.2363 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.xgb.equal.n25.dauroc_vs_src_L0` | -0.0521 | [-0.0643, -0.0411] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.aids.xgb.equal.n50.share` | 0.1374 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.xgb.equal.n50.dauroc_vs_src_L0` | -0.0293 | [-0.0400, -0.0207] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.aids.xgb.equal.nfull.share` | 0.0648 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.xgb.equal.nfull.dauroc_vs_src_L0` | -0.0123 | [-0.0174, -0.0082] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.xgb.dir0.5.n25.share` | 0.1333 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.xgb.dir0.5.n25.dauroc_vs_src_L0` | 0.0207 | [0.0030, 0.0366] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.xgb.dir0.5.n25.dauroc_vs_fed_L0` | -0.0424 | [-0.0607, -0.0259] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.aids.xgb.dir0.5.n50.share` | 0.0578 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.xgb.dir0.5.n50.dauroc_vs_src_L0` | 0.0440 | [0.0258, 0.0562] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.xgb.dir0.5.n50.dauroc_vs_fed_L0` | -0.0191 | [-0.0384, -0.0066] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.aids.xgb.dir0.5.nfull.share` | 0.0067 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.xgb.dir0.5.nfull.dauroc_vs_src_L0` | 0.0618 | [0.0585, 0.0653] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.xgb.dir0.5.nfull.dauroc_vs_fed_L0` | -0.0012 | [-0.0029, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.aids.xgb.dir0.5.n25.share` | 0.2893 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.xgb.dir0.5.n25.dauroc_vs_src_L0` | -0.0450 | [-0.0545, -0.0364] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.aids.xgb.dir0.5.n50.share` | 0.1967 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.xgb.dir0.5.n50.dauroc_vs_src_L0` | -0.0273 | [-0.0348, -0.0210] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.aids.xgb.dir0.5.nfull.share` | 0.1348 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.xgb.dir0.5.nfull.dauroc_vs_src_L0` | -0.0147 | [-0.0189, -0.0110] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.xgb.dir0.1.n25.share` | 0.2067 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.xgb.dir0.1.n25.dauroc_vs_src_L0` | 0.0117 | [-0.0029, 0.0254] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.xgb.dir0.1.n25.dauroc_vs_fed_L0` | -0.0513 | [-0.0673, -0.0373] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.aids.xgb.dir0.1.n50.share` | 0.1311 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.xgb.dir0.1.n50.dauroc_vs_src_L0` | 0.0327 | [0.0197, 0.0428] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.xgb.dir0.1.n50.dauroc_vs_fed_L0` | -0.0303 | [-0.0440, -0.0195] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.aids.xgb.dir0.1.nfull.share` | 0.0644 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.aids.xgb.dir0.1.nfull.dauroc_vs_src_L0` | 0.0520 | [0.0459, 0.0574] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.aids.xgb.dir0.1.nfull.dauroc_vs_fed_L0` | -0.0111 | [-0.0169, -0.0064] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.aids.xgb.dir0.1.n25.share` | 0.3430 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.xgb.dir0.1.n25.dauroc_vs_src_L0` | -0.0339 | [-0.0410, -0.0269] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.aids.xgb.dir0.1.n50.share` | 0.2900 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.xgb.dir0.1.n50.dauroc_vs_src_L0` | -0.0221 | [-0.0290, -0.0157] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.aids.xgb.dir0.1.nfull.share` | 0.2374 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.aids.xgb.dir0.1.nfull.dauroc_vs_src_L0` | -0.0110 | [-0.0147, -0.0074] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.rf.equal.n25.share` | 0.0156 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.diabetic.rf.equal.n25.dauroc_vs_src_L0` | 0.0331 | [0.0164, 0.0425] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.rf.equal.n25.dauroc_vs_fed_L0` | -0.0076 | [-0.0228, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.diabetic.rf.equal.nfull.share` | 0.0022 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.diabetic.rf.equal.nfull.dauroc_vs_src_L0` | 0.0395 | [0.0358, 0.0428] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.rf.equal.nfull.dauroc_vs_fed_L0` | -0.0011 | [-0.0033, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.diabetic.rf.equal.n25.share` | 0.0611 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.diabetic.rf.equal.n25.dauroc_vs_src_L0` | -0.0195 | [-0.0312, -0.0104] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.diabetic.rf.equal.nfull.share` | 0.0170 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.diabetic.rf.equal.nfull.dauroc_vs_src_L0` | -0.0045 | [-0.0092, -0.0014] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.rf.dir0.5.n25.share` | 0.0267 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.diabetic.rf.dir0.5.n25.dauroc_vs_src_L0` | 0.0518 | [0.0347, 0.0629] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.rf.dir0.5.n25.dauroc_vs_fed_L0` | -0.0112 | [-0.0275, -0.0009] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.diabetic.rf.dir0.5.nfull.share` | 0.0022 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.diabetic.rf.dir0.5.nfull.dauroc_vs_src_L0` | 0.0620 | [0.0580, 0.0658] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.rf.dir0.5.nfull.dauroc_vs_fed_L0` | -0.0010 | [-0.0031, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.diabetic.rf.dir0.5.n25.share` | 0.1470 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.diabetic.rf.dir0.5.n25.dauroc_vs_src_L0` | -0.0263 | [-0.0371, -0.0176] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.diabetic.rf.dir0.5.nfull.share` | 0.0807 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.diabetic.rf.dir0.5.nfull.dauroc_vs_src_L0` | -0.0088 | [-0.0141, -0.0047] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.rf.dir0.1.n25.share` | 0.0600 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.diabetic.rf.dir0.1.n25.dauroc_vs_src_L0` | 0.0454 | [0.0333, 0.0547] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.rf.dir0.1.n25.dauroc_vs_fed_L0` | -0.0171 | [-0.0293, -0.0079] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.diabetic.rf.dir0.1.nfull.share` | 0.0222 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.diabetic.rf.dir0.1.nfull.dauroc_vs_src_L0` | 0.0571 | [0.0506, 0.0627] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.rf.dir0.1.nfull.dauroc_vs_fed_L0` | -0.0055 | [-0.0102, -0.0017] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.diabetic.rf.dir0.1.n25.share` | 0.2352 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.diabetic.rf.dir0.1.n25.dauroc_vs_src_L0` | -0.0222 | [-0.0285, -0.0168] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.diabetic.rf.dir0.1.nfull.share` | 0.1678 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.diabetic.rf.dir0.1.nfull.dauroc_vs_src_L0` | -0.0093 | [-0.0138, -0.0056] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.xgb.equal.n25.share` | 0.0089 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.diabetic.xgb.equal.n25.dauroc_vs_src_L0` | 0.0634 | [0.0549, 0.0696] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.xgb.equal.n25.dauroc_vs_fed_L0` | -0.0049 | [-0.0134, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.diabetic.xgb.equal.nfull.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.diabetic.xgb.equal.nfull.dauroc_vs_src_L0` | 0.0683 | [0.0648, 0.0718] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.xgb.equal.nfull.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.diabetic.xgb.equal.n25.share` | 0.0689 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.diabetic.xgb.equal.n25.dauroc_vs_src_L0` | -0.0182 | [-0.0266, -0.0114] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.diabetic.xgb.equal.nfull.share` | 0.0259 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.diabetic.xgb.equal.nfull.dauroc_vs_src_L0` | -0.0059 | [-0.0102, -0.0028] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.xgb.dir0.5.n25.share` | 0.0133 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.diabetic.xgb.dir0.5.n25.dauroc_vs_src_L0` | 0.0820 | [0.0733, 0.0890] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.xgb.dir0.5.n25.dauroc_vs_fed_L0` | -0.0062 | [-0.0143, -0.0008] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.diabetic.xgb.dir0.5.nfull.share` | 0.0022 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.diabetic.xgb.dir0.5.nfull.dauroc_vs_src_L0` | 0.0871 | [0.0830, 0.0913] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.xgb.dir0.5.nfull.dauroc_vs_fed_L0` | -0.0010 | [-0.0031, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.diabetic.xgb.dir0.5.n25.share` | 0.1685 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.diabetic.xgb.dir0.5.n25.dauroc_vs_src_L0` | -0.0267 | [-0.0352, -0.0196] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.diabetic.xgb.dir0.5.nfull.share` | 0.0970 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.diabetic.xgb.dir0.5.nfull.dauroc_vs_src_L0` | -0.0098 | [-0.0134, -0.0066] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.xgb.dir0.1.n25.share` | 0.0489 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.diabetic.xgb.dir0.1.n25.dauroc_vs_src_L0` | 0.0810 | [0.0697, 0.0909] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.xgb.dir0.1.n25.dauroc_vs_fed_L0` | -0.0165 | [-0.0277, -0.0072] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.diabetic.xgb.dir0.1.nfull.share` | 0.0222 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.diabetic.xgb.dir0.1.nfull.dauroc_vs_src_L0` | 0.0915 | [0.0850, 0.0973] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.diabetic.xgb.dir0.1.nfull.dauroc_vs_fed_L0` | -0.0059 | [-0.0112, -0.0019] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.diabetic.xgb.dir0.1.n25.share` | 0.2737 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.diabetic.xgb.dir0.1.n25.dauroc_vs_src_L0` | -0.0203 | [-0.0263, -0.0146] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.diabetic.xgb.dir0.1.nfull.share` | 0.2111 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.diabetic.xgb.dir0.1.nfull.dauroc_vs_src_L0` | -0.0104 | [-0.0146, -0.0067] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.equal.n25.share` | 0.0289 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.rf.equal.n25.dauroc_vs_src_L0` | -0.0113 | [-0.0331, 0.0059] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.equal.n25.dauroc_vs_fed_L0` | -0.0187 | [-0.0406, -0.0014] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.rf.equal.n50.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.rf.equal.n50.dauroc_vs_src_L0` | 0.0074 | [0.0073, 0.0076] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.equal.n50.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.rf.equal.n100.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.rf.equal.n100.dauroc_vs_src_L0` | 0.0074 | [0.0073, 0.0076] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.equal.n100.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.rf.equal.n200.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.rf.equal.n200.dauroc_vs_src_L0` | 0.0074 | [0.0073, 0.0076] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.equal.n200.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.rf.equal.nfull.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.rf.equal.nfull.dauroc_vs_src_L0` | 0.0074 | [0.0073, 0.0076] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.equal.nfull.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.heart.rf.equal.n25.share` | 0.0419 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.rf.equal.n25.dauroc_vs_src_L0` | -0.0254 | [-0.0515, -0.0059] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.rf.equal.n50.share` | 0.0044 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.rf.equal.n50.dauroc_vs_src_L0` | -0.0026 | [-0.0060, -0.0003] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.rf.equal.n100.share` | 0.0004 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.rf.equal.n100.dauroc_vs_src_L0` | -0.0002 | [-0.0005, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.rf.equal.n200.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.rf.equal.n200.dauroc_vs_src_L0` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.rf.equal.nfull.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.rf.equal.nfull.dauroc_vs_src_L0` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.dir0.5.n25.share` | 0.0267 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.rf.dir0.5.n25.dauroc_vs_src_L0` | 0.0390 | [0.0121, 0.0554] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.dir0.5.n25.dauroc_vs_fed_L0` | -0.0175 | [-0.0439, -0.0014] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.rf.dir0.5.n50.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.rf.dir0.5.n50.dauroc_vs_src_L0` | 0.0565 | [0.0538, 0.0593] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.dir0.5.n50.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.rf.dir0.5.n100.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.rf.dir0.5.n100.dauroc_vs_src_L0` | 0.0565 | [0.0538, 0.0593] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.dir0.5.n100.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.rf.dir0.5.n200.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.rf.dir0.5.n200.dauroc_vs_src_L0` | 0.0565 | [0.0538, 0.0593] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.dir0.5.n200.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.rf.dir0.5.nfull.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.rf.dir0.5.nfull.dauroc_vs_src_L0` | 0.0565 | [0.0538, 0.0593] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.dir0.5.nfull.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.heart.rf.dir0.5.n25.share` | 0.1126 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.rf.dir0.5.n25.dauroc_vs_src_L0` | -0.0391 | [-0.0617, -0.0243] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.rf.dir0.5.n50.share` | 0.0615 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.rf.dir0.5.n50.dauroc_vs_src_L0` | -0.0128 | [-0.0174, -0.0086] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.rf.dir0.5.n100.share` | 0.0381 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.rf.dir0.5.n100.dauroc_vs_src_L0` | -0.0060 | [-0.0083, -0.0039] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.rf.dir0.5.n200.share` | 0.0215 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.rf.dir0.5.n200.dauroc_vs_src_L0` | -0.0023 | [-0.0033, -0.0015] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.rf.dir0.5.nfull.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.rf.dir0.5.nfull.dauroc_vs_src_L0` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.dir0.1.n25.share` | 0.0622 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.rf.dir0.1.n25.dauroc_vs_src_L0` | 0.0719 | [0.0486, 0.0911] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.dir0.1.n25.dauroc_vs_fed_L0` | -0.0322 | [-0.0545, -0.0141] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.rf.dir0.1.n50.share` | 0.0178 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.rf.dir0.1.n50.dauroc_vs_src_L0` | 0.0963 | [0.0874, 0.1034] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.dir0.1.n50.dauroc_vs_fed_L0` | -0.0078 | [-0.0150, -0.0023] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.rf.dir0.1.n100.share` | 0.0044 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.rf.dir0.1.n100.dauroc_vs_src_L0` | 0.1034 | [0.1002, 0.1064] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.dir0.1.n100.dauroc_vs_fed_L0` | -0.0007 | [-0.0019, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.rf.dir0.1.n200.share` | 0.0022 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.rf.dir0.1.n200.dauroc_vs_src_L0` | 0.1037 | [0.1005, 0.1068] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.dir0.1.n200.dauroc_vs_fed_L0` | -0.0004 | [-0.0012, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.rf.dir0.1.nfull.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.rf.dir0.1.nfull.dauroc_vs_src_L0` | 0.1041 | [0.1010, 0.1071] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.rf.dir0.1.nfull.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.heart.rf.dir0.1.n25.share` | 0.1856 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.rf.dir0.1.n25.dauroc_vs_src_L0` | -0.0449 | [-0.0610, -0.0331] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.rf.dir0.1.n50.share` | 0.1504 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.rf.dir0.1.n50.dauroc_vs_src_L0` | -0.0255 | [-0.0332, -0.0192] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.rf.dir0.1.n100.share` | 0.1081 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.rf.dir0.1.n100.dauroc_vs_src_L0` | -0.0149 | [-0.0186, -0.0115] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.rf.dir0.1.n200.share` | 0.0637 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.rf.dir0.1.n200.dauroc_vs_src_L0` | -0.0048 | [-0.0060, -0.0037] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.rf.dir0.1.nfull.share` | 0.0037 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.rf.dir0.1.nfull.dauroc_vs_src_L0` | 0.0003 | [0.0000, 0.0006] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.equal.n25.share` | 0.0200 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.xgb.equal.n25.dauroc_vs_src_L0` | 0.0156 | [-0.0119, 0.0294] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.equal.n25.dauroc_vs_fed_L0` | -0.0137 | [-0.0411, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.xgb.equal.n50.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.xgb.equal.n50.dauroc_vs_src_L0` | 0.0293 | [0.0291, 0.0294] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.equal.n50.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.xgb.equal.n100.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.xgb.equal.n100.dauroc_vs_src_L0` | 0.0293 | [0.0291, 0.0294] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.equal.n100.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.xgb.equal.n200.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.xgb.equal.n200.dauroc_vs_src_L0` | 0.0293 | [0.0291, 0.0294] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.equal.n200.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.xgb.equal.nfull.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.xgb.equal.nfull.dauroc_vs_src_L0` | 0.0293 | [0.0291, 0.0294] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.equal.nfull.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.heart.xgb.equal.n25.share` | 0.0226 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.xgb.equal.n25.dauroc_vs_src_L0` | -0.0140 | [-0.0347, -0.0013] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.xgb.equal.n50.share` | 0.0022 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.xgb.equal.n50.dauroc_vs_src_L0` | -0.0009 | [-0.0018, -0.0002] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.xgb.equal.n100.share` | 0.0007 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.xgb.equal.n100.dauroc_vs_src_L0` | -0.0003 | [-0.0008, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.xgb.equal.n200.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.xgb.equal.n200.dauroc_vs_src_L0` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.xgb.equal.nfull.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.xgb.equal.nfull.dauroc_vs_src_L0` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.dir0.5.n25.share` | 0.0200 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.xgb.dir0.5.n25.dauroc_vs_src_L0` | 0.0352 | [0.0099, 0.0490] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.dir0.5.n25.dauroc_vs_fed_L0` | -0.0133 | [-0.0386, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.xgb.dir0.5.n50.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.xgb.dir0.5.n50.dauroc_vs_src_L0` | 0.0485 | [0.0473, 0.0497] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.dir0.5.n50.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.xgb.dir0.5.n100.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.xgb.dir0.5.n100.dauroc_vs_src_L0` | 0.0485 | [0.0473, 0.0497] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.dir0.5.n100.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.xgb.dir0.5.n200.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.xgb.dir0.5.n200.dauroc_vs_src_L0` | 0.0485 | [0.0473, 0.0497] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.dir0.5.n200.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.xgb.dir0.5.nfull.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.xgb.dir0.5.nfull.dauroc_vs_src_L0` | 0.0485 | [0.0473, 0.0497] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.dir0.5.nfull.dauroc_vs_fed_L0` | 0.0000 | [0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.heart.xgb.dir0.5.n25.share` | 0.0685 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.xgb.dir0.5.n25.dauroc_vs_src_L0` | -0.0295 | [-0.0470, -0.0178] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.xgb.dir0.5.n50.share` | 0.0252 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.xgb.dir0.5.n50.dauroc_vs_src_L0` | -0.0096 | [-0.0138, -0.0064] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.xgb.dir0.5.n100.share` | 0.0148 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.xgb.dir0.5.n100.dauroc_vs_src_L0` | -0.0055 | [-0.0078, -0.0033] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.xgb.dir0.5.n200.share` | 0.0041 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.xgb.dir0.5.n200.dauroc_vs_src_L0` | -0.0012 | [-0.0022, -0.0004] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.xgb.dir0.5.nfull.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.xgb.dir0.5.nfull.dauroc_vs_src_L0` | 0.0000 | [0.0000, 0.0000] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.dir0.1.n25.share` | 0.0489 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.xgb.dir0.1.n25.dauroc_vs_src_L0` | 0.0628 | [0.0335, 0.0818] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.dir0.1.n25.dauroc_vs_fed_L0` | -0.0292 | [-0.0590, -0.0102] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.xgb.dir0.1.n50.share` | 0.0044 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.xgb.dir0.1.n50.dauroc_vs_src_L0` | 0.0901 | [0.0870, 0.0927] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.dir0.1.n50.dauroc_vs_fed_L0` | -0.0019 | [-0.0052, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.xgb.dir0.1.n100.share` | 0.0022 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.xgb.dir0.1.n100.dauroc_vs_src_L0` | 0.0914 | [0.0895, 0.0933] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.dir0.1.n100.dauroc_vs_fed_L0` | -0.0006 | [-0.0018, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.xgb.dir0.1.n200.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.xgb.dir0.1.n200.dauroc_vs_src_L0` | 0.0920 | [0.0904, 0.0937] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.dir0.1.n200.dauroc_vs_fed_L0` | 0.0000 | [-0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.fed_L2.heart.xgb.dir0.1.nfull.share` | 0.0000 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.fed_L2.heart.xgb.dir0.1.nfull.dauroc_vs_src_L0` | 0.0920 | [0.0904, 0.0937] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.fed_L2.heart.xgb.dir0.1.nfull.dauroc_vs_fed_L0` | 0.0000 | [-0.0000, 0.0000] | 50.0 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. fed_L2 minus fed_L0. |
| `l2neg.src_L2.heart.xgb.dir0.1.n25.share` | 0.1874 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.xgb.dir0.1.n25.dauroc_vs_src_L0` | -0.0558 | [-0.0714, -0.0428] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.xgb.dir0.1.n50.share` | 0.1133 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.xgb.dir0.1.n50.dauroc_vs_src_L0` | -0.0283 | [-0.0352, -0.0221] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.xgb.dir0.1.n100.share` | 0.0807 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.xgb.dir0.1.n100.dauroc_vs_src_L0` | -0.0163 | [-0.0207, -0.0121] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.xgb.dir0.1.n200.share` | 0.0404 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.xgb.dir0.1.n200.dauroc_vs_src_L0` | -0.0054 | [-0.0073, -0.0036] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `l2neg.src_L2.heart.xgb.dir0.1.nfull.share` | 0.0104 |  |  | Share of level 2 fits with a <= 0, pooled over K, draws and sources (eval test, default configuration). |
| `l2neg.src_L2.heart.xgb.dir0.1.nfull.dauroc_vs_src_L0` | 0.0007 | [0.0003, 0.0011] | 50 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Level 2 arm minus src_L0 (mean source model). |
| `e12.r.brier.L2.aids.rf.equal.n25` | -808.9507 | [-57.2261, 60.2765] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: no. |
| `e12.r.brier.L2.aids.rf.equal.n50` | -0.2497 | [-1.4450, 0.0903] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.rf.equal.nfull` | 0.1772 | [0.0471, 0.2823] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.rf.equal.n100` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.aids.rf.equal.n200` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.aids.rf.dir0.5.n25` | 0.9292 | [0.9157, 0.9425] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.rf.dir0.5.n50` | 0.9397 | [0.9310, 0.9481] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.rf.dir0.5.nfull` | 0.9411 | [0.9335, 0.9484] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.rf.dir0.5.n100` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.aids.rf.dir0.5.n200` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.aids.rf.dir0.1.n25` | 0.9746 | [0.9648, 0.9851] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.rf.dir0.1.n50` | 0.9776 | [0.9729, 0.9821] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.rf.dir0.1.nfull` | 0.9763 | [0.9730, 0.9795] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.rf.dir0.1.n100` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.aids.rf.dir0.1.n200` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.aids.xgb.equal.n25` | 0.7858 | [0.7550, 0.8187] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.xgb.equal.n50` | 0.8315 | [0.8052, 0.8591] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.xgb.equal.nfull` | 0.8307 | [0.8092, 0.8509] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.xgb.equal.n100` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.aids.xgb.equal.n200` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.aids.xgb.dir0.5.n25` | 0.9389 | [0.9305, 0.9466] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.xgb.dir0.5.n50` | 0.9490 | [0.9420, 0.9567] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.xgb.dir0.5.nfull` | 0.9482 | [0.9427, 0.9537] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.xgb.dir0.5.n100` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.aids.xgb.dir0.5.n200` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.aids.xgb.dir0.1.n25` | 0.9709 | [0.9643, 0.9775] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.xgb.dir0.1.n50` | 0.9754 | [0.9714, 0.9795] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.xgb.dir0.1.nfull` | 0.9737 | [0.9708, 0.9767] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.aids.xgb.dir0.1.n100` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.aids.xgb.dir0.1.n200` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.rf.equal.n25` | -1.2682 | [-4.0035, -0.6354] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.diabetic.rf.equal.nfull` | -0.2126 | [-0.4175, -0.0861] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.diabetic.rf.equal.n50` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.rf.equal.n100` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.rf.equal.n200` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.rf.dir0.5.n25` | 0.7917 | [0.7691, 0.8135] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.diabetic.rf.dir0.5.nfull` | 0.8092 | [0.7936, 0.8248] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.diabetic.rf.dir0.5.n50` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.rf.dir0.5.n100` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.rf.dir0.5.n200` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.rf.dir0.1.n25` | 0.8986 | [0.8813, 0.9170] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.diabetic.rf.dir0.1.nfull` | 0.9108 | [0.8997, 0.9235] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.diabetic.rf.dir0.1.n50` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.rf.dir0.1.n100` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.rf.dir0.1.n200` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.xgb.equal.n25` | 0.6278 | [0.5954, 0.6597] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.diabetic.xgb.equal.nfull` | 0.6532 | [0.6243, 0.6830] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.diabetic.xgb.equal.n50` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.xgb.equal.n100` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.xgb.equal.n200` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.xgb.dir0.5.n25` | 0.8140 | [0.7975, 0.8294] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.diabetic.xgb.dir0.5.nfull` | 0.8288 | [0.8151, 0.8429] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.diabetic.xgb.dir0.5.n50` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.xgb.dir0.5.n100` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.xgb.dir0.5.n200` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.xgb.dir0.1.n25` | 0.8652 | [0.8507, 0.8783] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.diabetic.xgb.dir0.1.nfull` | 0.8757 | [0.8649, 0.8867] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.diabetic.xgb.dir0.1.n50` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.xgb.dir0.1.n100` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.diabetic.xgb.dir0.1.n200` | null |  | 0 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Null: not computable from the run files. Reportable: no. |
| `e12.r.brier.L2.heart.rf.equal.n25` | 1.1629 | [1.0692, 1.3084] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.rf.equal.n50` | 1.1539 | [0.9874, 1.5475] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.rf.equal.n100` | 1.5812 | [1.2738, 2.5493] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.rf.equal.n200` | 14.8455 | [-63.8352, 49.7885] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: no. |
| `e12.r.brier.L2.heart.rf.equal.nfull` | 0.3022 | [0.2833, 0.3195] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.rf.dir0.5.n25` | 0.9772 | [0.9592, 0.9994] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.rf.dir0.5.n50` | 0.9718 | [0.9633, 0.9799] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.rf.dir0.5.n100` | 0.9766 | [0.9723, 0.9805] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.rf.dir0.5.n200` | 0.9781 | [0.9751, 0.9806] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.rf.dir0.5.nfull` | 0.9790 | [0.9768, 0.9809] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.rf.dir0.1.n25` | 1.0041 | [0.9962, 1.0125] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.rf.dir0.1.n50` | 0.9949 | [0.9927, 0.9975] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.rf.dir0.1.n100` | 0.9931 | [0.9921, 0.9941] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.rf.dir0.1.n200` | 0.9929 | [0.9921, 0.9936] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.rf.dir0.1.nfull` | 0.9926 | [0.9921, 0.9930] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.xgb.equal.n25` | 1.3689 | [1.0446, 2.4045] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.xgb.equal.n50` | -2.6288 | [-22.3165, 28.6561] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: no. |
| `e12.r.brier.L2.heart.xgb.equal.n100` | 0.2004 | [0.0229, 0.3004] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.xgb.equal.n200` | 0.4066 | [0.3695, 0.4372] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.xgb.equal.nfull` | 0.5162 | [0.5096, 0.5231] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.xgb.dir0.5.n25` | 0.9746 | [0.9539, 1.0030] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.xgb.dir0.5.n50` | 0.9719 | [0.9656, 0.9795] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.xgb.dir0.5.n100` | 0.9773 | [0.9740, 0.9808] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.xgb.dir0.5.n200` | 0.9765 | [0.9739, 0.9788] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.xgb.dir0.5.nfull` | 0.9755 | [0.9730, 0.9776] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.xgb.dir0.1.n25` | 0.9924 | [0.9857, 0.9997] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.xgb.dir0.1.n50` | 0.9942 | [0.9913, 0.9975] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.xgb.dir0.1.n100` | 0.9932 | [0.9920, 0.9946] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.xgb.dir0.1.n200` | 0.9925 | [0.9918, 0.9932] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `e12.r.brier.L2.heart.xgb.dir0.1.nfull` | 0.9919 | [0.9915, 0.9923] | 50 | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. src_L2 and fed_L2 fitted on n_cal calibration records. Reportable: yes. |
| `hp.default.aids.rf.equal.r.brier.L1` | 0.0001 | [0.0000, 0.0002] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.aids.rf.equal.r.brier.L2` | 0.1150 | [-0.1280, 0.2610] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.aids.rf.equal.diff.auroc.L3` | 0.0341 | [0.0309, 0.0374] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.aids.rf.equal.auroc.central` | 0.6580 |  |  | Mean over the common draws. |
| `hp.default.aids.rf.equal.brier.central` | 0.1744 |  |  | Mean over the common draws. |
| `hp.regularized.aids.rf.equal.r.brier.L1` | 0.0004 | [0.0003, 0.0006] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.aids.rf.equal.r.brier.L2` | -0.7995 | [-2.1950, -0.3284] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.aids.rf.equal.diff.auroc.L3` | 0.0294 | [0.0271, 0.0317] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.aids.rf.equal.auroc.central` | 0.6775 |  |  | Mean over the common draws. |
| `hp.regularized.aids.rf.equal.brier.central` | 0.1689 |  |  | Mean over the common draws. |
| `hp.default.aids.rf.dir0.5.r.brier.L1` | 0.8057 | [0.7906, 0.8195] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.aids.rf.dir0.5.r.brier.L2` | 0.9402 | [0.9299, 0.9506] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.aids.rf.dir0.5.diff.auroc.L3` | 0.0451 | [0.0409, 0.0499] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.aids.rf.dir0.5.auroc.central` | 0.6580 |  |  | Mean over the common draws. |
| `hp.default.aids.rf.dir0.5.brier.central` | 0.1744 |  |  | Mean over the common draws. |
| `hp.regularized.aids.rf.dir0.5.r.brier.L1` | 0.9117 | [0.9031, 0.9192] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.aids.rf.dir0.5.r.brier.L2` | 0.9329 | [0.9230, 0.9421] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.aids.rf.dir0.5.diff.auroc.L3` | 0.0460 | [0.0418, 0.0508] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.aids.rf.dir0.5.auroc.central` | 0.6775 |  |  | Mean over the common draws. |
| `hp.regularized.aids.rf.dir0.5.brier.central` | 0.1689 |  |  | Mean over the common draws. |
| `hp.default.aids.rf.dir0.1.r.brier.L1` | 0.7260 | [0.7108, 0.7430] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.aids.rf.dir0.1.r.brier.L2` | 0.9752 | [0.9710, 0.9789] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.aids.rf.dir0.1.diff.auroc.L3` | 0.0481 | [0.0439, 0.0523] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.aids.rf.dir0.1.auroc.central` | 0.6580 |  |  | Mean over the common draws. |
| `hp.default.aids.rf.dir0.1.brier.central` | 0.1744 |  |  | Mean over the common draws. |
| `hp.regularized.aids.rf.dir0.1.r.brier.L1` | 0.8980 | [0.8915, 0.9044] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.aids.rf.dir0.1.r.brier.L2` | 0.9675 | [0.9615, 0.9729] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.aids.rf.dir0.1.diff.auroc.L3` | 0.0632 | [0.0565, 0.0702] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.aids.rf.dir0.1.auroc.central` | 0.6775 |  |  | Mean over the common draws. |
| `hp.regularized.aids.rf.dir0.1.brier.central` | 0.1689 |  |  | Mean over the common draws. |
| `hp.default.aids.xgb.equal.r.brier.L1` | 0.0000 | [-0.0000, 0.0001] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.aids.xgb.equal.r.brier.L2` | 0.8288 | [0.8021, 0.8540] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.aids.xgb.equal.diff.auroc.L3` | 0.0590 | [0.0540, 0.0639] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.aids.xgb.equal.auroc.central` | 0.6203 |  |  | Mean over the common draws. |
| `hp.default.aids.xgb.equal.brier.central` | 0.2050 |  |  | Mean over the common draws. |
| `hp.regularized.aids.xgb.equal.r.brier.L1` | 0.0001 | [0.0000, 0.0002] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.aids.xgb.equal.r.brier.L2` | 0.7372 | [0.6992, 0.7741] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.aids.xgb.equal.diff.auroc.L3` | 0.0579 | [0.0530, 0.0624] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.aids.xgb.equal.auroc.central` | 0.6442 |  |  | Mean over the common draws. |
| `hp.regularized.aids.xgb.equal.brier.central` | 0.1825 |  |  | Mean over the common draws. |
| `hp.default.aids.xgb.dir0.5.r.brier.L1` | 0.4788 | [0.4662, 0.4914] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.aids.xgb.dir0.5.r.brier.L2` | 0.9486 | [0.9416, 0.9557] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.aids.xgb.dir0.5.diff.auroc.L3` | 0.0633 | [0.0581, 0.0681] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.aids.xgb.dir0.5.auroc.central` | 0.6203 |  |  | Mean over the common draws. |
| `hp.default.aids.xgb.dir0.5.brier.central` | 0.2050 |  |  | Mean over the common draws. |
| `hp.regularized.aids.xgb.dir0.5.r.brier.L1` | 0.6144 | [0.6013, 0.6268] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.aids.xgb.dir0.5.r.brier.L2` | 0.9439 | [0.9359, 0.9520] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.aids.xgb.dir0.5.diff.auroc.L3` | 0.0608 | [0.0558, 0.0656] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.aids.xgb.dir0.5.auroc.central` | 0.6442 |  |  | Mean over the common draws. |
| `hp.regularized.aids.xgb.dir0.5.brier.central` | 0.1825 |  |  | Mean over the common draws. |
| `hp.default.aids.xgb.dir0.1.r.brier.L1` | 0.5993 | [0.5900, 0.6084] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.aids.xgb.dir0.1.r.brier.L2` | 0.9755 | [0.9718, 0.9793] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.aids.xgb.dir0.1.diff.auroc.L3` | 0.0634 | [0.0584, 0.0688] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.aids.xgb.dir0.1.auroc.central` | 0.6203 |  |  | Mean over the common draws. |
| `hp.default.aids.xgb.dir0.1.brier.central` | 0.2050 |  |  | Mean over the common draws. |
| `hp.regularized.aids.xgb.dir0.1.r.brier.L1` | 0.7307 | [0.7224, 0.7382] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.aids.xgb.dir0.1.r.brier.L2` | 0.9728 | [0.9688, 0.9769] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.aids.xgb.dir0.1.diff.auroc.L3` | 0.0633 | [0.0583, 0.0688] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.aids.xgb.dir0.1.auroc.central` | 0.6442 |  |  | Mean over the common draws. |
| `hp.regularized.aids.xgb.dir0.1.brier.central` | 0.1825 |  |  | Mean over the common draws. |
| `hp.default.diabetic.rf.equal.r.brier.L1` | 0.0002 | [0.0000, 0.0003] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.diabetic.rf.equal.r.brier.L2` | -0.2431 | [-0.8098, -0.0387] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.diabetic.rf.equal.diff.auroc.L3` | 0.0421 | [0.0374, 0.0467] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.diabetic.rf.equal.auroc.central` | 0.7559 |  |  | Mean over the common draws. |
| `hp.default.diabetic.rf.equal.brier.central` | 0.1980 |  |  | Mean over the common draws. |
| `hp.regularized.diabetic.rf.equal.r.brier.L1` | 0.0002 | [-0.0000, 0.0003] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.diabetic.rf.equal.r.brier.L2` | -0.6421 | [-3.1630, -0.2026] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.diabetic.rf.equal.diff.auroc.L3` | 0.0326 | [0.0282, 0.0371] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.diabetic.rf.equal.auroc.central` | 0.7595 |  |  | Mean over the common draws. |
| `hp.regularized.diabetic.rf.equal.brier.central` | 0.1975 |  |  | Mean over the common draws. |
| `hp.default.diabetic.rf.dir0.5.r.brier.L1` | 0.6291 | [0.6164, 0.6410] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.diabetic.rf.dir0.5.r.brier.L2` | 0.8007 | [0.7836, 0.8186] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.diabetic.rf.dir0.5.diff.auroc.L3` | 0.0633 | [0.0572, 0.0687] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.diabetic.rf.dir0.5.auroc.central` | 0.7559 |  |  | Mean over the common draws. |
| `hp.default.diabetic.rf.dir0.5.brier.central` | 0.1980 |  |  | Mean over the common draws. |
| `hp.regularized.diabetic.rf.dir0.5.r.brier.L1` | 0.7772 | [0.7647, 0.7879] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.diabetic.rf.dir0.5.r.brier.L2` | 0.8366 | [0.8104, 0.8670] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.diabetic.rf.dir0.5.diff.auroc.L3` | 0.0592 | [0.0525, 0.0651] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.diabetic.rf.dir0.5.auroc.central` | 0.7595 |  |  | Mean over the common draws. |
| `hp.regularized.diabetic.rf.dir0.5.brier.central` | 0.1975 |  |  | Mean over the common draws. |
| `hp.default.diabetic.rf.dir0.1.r.brier.L1` | 0.5318 | [0.5148, 0.5464] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.diabetic.rf.dir0.1.r.brier.L2` | 0.9078 | [0.8936, 0.9210] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.diabetic.rf.dir0.1.diff.auroc.L3` | 0.0602 | [0.0548, 0.0653] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.diabetic.rf.dir0.1.auroc.central` | 0.7559 |  |  | Mean over the common draws. |
| `hp.default.diabetic.rf.dir0.1.brier.central` | 0.1980 |  |  | Mean over the common draws. |
| `hp.regularized.diabetic.rf.dir0.1.r.brier.L1` | 0.7535 | [0.7401, 0.7651] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.diabetic.rf.dir0.1.r.brier.L2` | 0.8736 | [0.8561, 0.8909] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.diabetic.rf.dir0.1.diff.auroc.L3` | 0.0894 | [0.0816, 0.0975] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.diabetic.rf.dir0.1.auroc.central` | 0.7595 |  |  | Mean over the common draws. |
| `hp.regularized.diabetic.rf.dir0.1.brier.central` | 0.1975 |  |  | Mean over the common draws. |
| `hp.default.diabetic.xgb.equal.r.brier.L1` | 0.0000 | [-0.0000, 0.0000] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.diabetic.xgb.equal.r.brier.L2` | 0.6473 | [0.5991, 0.7084] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.diabetic.xgb.equal.diff.auroc.L3` | 0.0693 | [0.0626, 0.0755] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.diabetic.xgb.equal.auroc.central` | 0.7771 |  |  | Mean over the common draws. |
| `hp.default.diabetic.xgb.equal.brier.central` | 0.2136 |  |  | Mean over the common draws. |
| `hp.regularized.diabetic.xgb.equal.r.brier.L1` | 0.0000 | [-0.0000, 0.0001] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.diabetic.xgb.equal.r.brier.L2` | 0.5486 | [0.4898, 0.6207] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.diabetic.xgb.equal.diff.auroc.L3` | 0.0659 | [0.0596, 0.0717] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.diabetic.xgb.equal.auroc.central` | 0.7760 |  |  | Mean over the common draws. |
| `hp.regularized.diabetic.xgb.equal.brier.central` | 0.1961 |  |  | Mean over the common draws. |
| `hp.default.diabetic.xgb.dir0.5.r.brier.L1` | 0.3485 | [0.3336, 0.3616] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.diabetic.xgb.dir0.5.r.brier.L2` | 0.8283 | [0.8065, 0.8535] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.diabetic.xgb.dir0.5.diff.auroc.L3` | 0.0890 | [0.0816, 0.0958] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.diabetic.xgb.dir0.5.auroc.central` | 0.7771 |  |  | Mean over the common draws. |
| `hp.default.diabetic.xgb.dir0.5.brier.central` | 0.2136 |  |  | Mean over the common draws. |
| `hp.regularized.diabetic.xgb.dir0.5.r.brier.L1` | 0.4558 | [0.4389, 0.4702] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.diabetic.xgb.dir0.5.r.brier.L2` | 0.8262 | [0.8032, 0.8522] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.diabetic.xgb.dir0.5.diff.auroc.L3` | 0.0832 | [0.0758, 0.0901] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.diabetic.xgb.dir0.5.auroc.central` | 0.7760 |  |  | Mean over the common draws. |
| `hp.regularized.diabetic.xgb.dir0.5.brier.central` | 0.1961 |  |  | Mean over the common draws. |
| `hp.default.diabetic.xgb.dir0.1.r.brier.L1` | 0.4167 | [0.4033, 0.4306] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.diabetic.xgb.dir0.1.r.brier.L2` | 0.8705 | [0.8542, 0.8867] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.diabetic.xgb.dir0.1.diff.auroc.L3` | 0.0993 | [0.0939, 0.1050] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.diabetic.xgb.dir0.1.auroc.central` | 0.7771 |  |  | Mean over the common draws. |
| `hp.default.diabetic.xgb.dir0.1.brier.central` | 0.2136 |  |  | Mean over the common draws. |
| `hp.regularized.diabetic.xgb.dir0.1.r.brier.L1` | 0.5356 | [0.5218, 0.5500] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.diabetic.xgb.dir0.1.r.brier.L2` | 0.8728 | [0.8554, 0.8909] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.diabetic.xgb.dir0.1.diff.auroc.L3` | 0.0937 | [0.0882, 0.0995] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.diabetic.xgb.dir0.1.auroc.central` | 0.7760 |  |  | Mean over the common draws. |
| `hp.regularized.diabetic.xgb.dir0.1.brier.central` | 0.1961 |  |  | Mean over the common draws. |
| `hp.default.heart.rf.equal.r.brier.L1` | 0.0000 | [-0.0000, 0.0000] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.heart.rf.equal.r.brier.L2` | 0.3040 | [0.2720, 0.3339] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.heart.rf.equal.diff.auroc.L3` | 0.0075 | [0.0072, 0.0077] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.heart.rf.equal.auroc.central` | 0.8251 |  |  | Mean over the common draws. |
| `hp.default.heart.rf.equal.brier.central` | 0.0464 |  |  | Mean over the common draws. |
| `hp.regularized.heart.rf.equal.r.brier.L1` | 0.0000 | [-0.0000, 0.0000] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.heart.rf.equal.r.brier.L2` | 0.1190 | [0.0892, 0.1437] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.heart.rf.equal.diff.auroc.L3` | 0.0047 | [0.0046, 0.0048] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.heart.rf.equal.auroc.central` | 0.8386 |  |  | Mean over the common draws. |
| `hp.regularized.heart.rf.equal.brier.central` | 0.0457 |  |  | Mean over the common draws. |
| `hp.default.heart.rf.dir0.5.r.brier.L1` | 0.9113 | [0.8803, 0.9368] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.heart.rf.dir0.5.r.brier.L2` | 0.9781 | [0.9746, 0.9807] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.heart.rf.dir0.5.diff.auroc.L3` | 0.0583 | [0.0533, 0.0636] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.heart.rf.dir0.5.auroc.central` | 0.8251 |  |  | Mean over the common draws. |
| `hp.default.heart.rf.dir0.5.brier.central` | 0.0464 |  |  | Mean over the common draws. |
| `hp.regularized.heart.rf.dir0.5.r.brier.L1` | 0.9642 | [0.9478, 0.9760] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.heart.rf.dir0.5.r.brier.L2` | 0.9810 | [0.9776, 0.9834] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.heart.rf.dir0.5.diff.auroc.L3` | 0.0377 | [0.0335, 0.0421] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.heart.rf.dir0.5.auroc.central` | 0.8386 |  |  | Mean over the common draws. |
| `hp.regularized.heart.rf.dir0.5.brier.central` | 0.0457 |  |  | Mean over the common draws. |
| `hp.default.heart.rf.dir0.1.r.brier.L1` | 0.7188 | [0.6866, 0.7505] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.heart.rf.dir0.1.r.brier.L2` | 0.9923 | [0.9916, 0.9930] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.heart.rf.dir0.1.diff.auroc.L3` | 0.1044 | [0.0986, 0.1101] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.heart.rf.dir0.1.auroc.central` | 0.8251 |  |  | Mean over the common draws. |
| `hp.default.heart.rf.dir0.1.brier.central` | 0.0464 |  |  | Mean over the common draws. |
| `hp.regularized.heart.rf.dir0.1.r.brier.L1` | 0.8622 | [0.8428, 0.8807] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.heart.rf.dir0.1.r.brier.L2` | 0.9902 | [0.9893, 0.9911] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.heart.rf.dir0.1.diff.auroc.L3` | 0.1167 | [0.1116, 0.1220] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.heart.rf.dir0.1.auroc.central` | 0.8386 |  |  | Mean over the common draws. |
| `hp.regularized.heart.rf.dir0.1.brier.central` | 0.0457 |  |  | Mean over the common draws. |
| `hp.default.heart.xgb.equal.r.brier.L1` | 0.0000 | [-0.0000, 0.0000] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.heart.xgb.equal.r.brier.L2` | 0.5127 | [0.5013, 0.5242] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.heart.xgb.equal.diff.auroc.L3` | 0.0293 | [0.0290, 0.0295] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.heart.xgb.equal.auroc.central` | 0.8370 |  |  | Mean over the common draws. |
| `hp.default.heart.xgb.equal.brier.central` | 0.0464 |  |  | Mean over the common draws. |
| `hp.regularized.heart.xgb.equal.r.brier.L1` | -0.0000 | [-0.0000, 0.0000] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.heart.xgb.equal.r.brier.L2` | 0.2045 | [0.1837, 0.2262] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.heart.xgb.equal.diff.auroc.L3` | 0.0115 | [0.0114, 0.0116] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.heart.xgb.equal.auroc.central` | 0.8463 |  |  | Mean over the common draws. |
| `hp.regularized.heart.xgb.equal.brier.central` | 0.0454 |  |  | Mean over the common draws. |
| `hp.default.heart.xgb.dir0.5.r.brier.L1` | 0.8150 | [0.8012, 0.8292] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.heart.xgb.dir0.5.r.brier.L2` | 0.9753 | [0.9718, 0.9781] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.heart.xgb.dir0.5.diff.auroc.L3` | 0.0489 | [0.0466, 0.0513] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.heart.xgb.dir0.5.auroc.central` | 0.8370 |  |  | Mean over the common draws. |
| `hp.default.heart.xgb.dir0.5.brier.central` | 0.0464 |  |  | Mean over the common draws. |
| `hp.regularized.heart.xgb.dir0.5.r.brier.L1` | 0.9360 | [0.9291, 0.9421] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.heart.xgb.dir0.5.r.brier.L2` | 0.9800 | [0.9770, 0.9823] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.heart.xgb.dir0.5.diff.auroc.L3` | 0.0320 | [0.0297, 0.0345] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.heart.xgb.dir0.5.auroc.central` | 0.8463 |  |  | Mean over the common draws. |
| `hp.regularized.heart.xgb.dir0.5.brier.central` | 0.0454 |  |  | Mean over the common draws. |
| `hp.default.heart.xgb.dir0.1.r.brier.L1` | 0.8282 | [0.8191, 0.8378] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.heart.xgb.dir0.1.r.brier.L2` | 0.9918 | [0.9912, 0.9923] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.default.heart.xgb.dir0.1.diff.auroc.L3` | 0.0926 | [0.0900, 0.0950] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.default.heart.xgb.dir0.1.auroc.central` | 0.8370 |  |  | Mean over the common draws. |
| `hp.default.heart.xgb.dir0.1.brier.central` | 0.0464 |  |  | Mean over the common draws. |
| `hp.regularized.heart.xgb.dir0.1.r.brier.L1` | 0.9431 | [0.9399, 0.9463] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.heart.xgb.dir0.1.r.brier.L2` | 0.9920 | [0.9914, 0.9925] |  | r_l = mean(m_l - m0) / mean(m3l - m0), K averaged, bootstrap interval of the ratio; reportable when the interval of mean(m3l - m0) excludes 0. Common draws. Reportable: yes. |
| `hp.regularized.heart.xgb.dir0.1.diff.auroc.L3` | 0.0848 | [0.0821, 0.0873] | 20 | Mean over draws of the paired difference, after averaging over K = 2..10 with equal weight. Common draws. |
| `hp.regularized.heart.xgb.dir0.1.auroc.central` | 0.8463 |  |  | Mean over the common draws. |
| `hp.regularized.heart.xgb.dir0.1.brier.central` | 0.0454 |  |  | Mean over the common draws. |
| `hp.default.leaf.rf.central.central.aids.single` | 0.5125 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.rf.central.central.aids.lt5` | 0.8417 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.xgb.central.central.aids.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.xgb.central.central.aids.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.rf.site.all.aids.single` | 0.4988 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.rf.site.all.aids.lt5` | 0.8212 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.all.aids.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.all.aids.lt5` | 0.0001 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.rf.site.dir0.1.aids.single` | 0.4868 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.rf.site.dir0.1.aids.lt5` | 0.7915 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.dir0.1.aids.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.dir0.1.aids.lt5` | 0.0003 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.rf.site.dir0.5.aids.single` | 0.4964 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.rf.site.dir0.5.aids.lt5` | 0.8133 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.dir0.5.aids.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.dir0.5.aids.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.rf.site.equal.aids.single` | 0.5055 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.rf.site.equal.aids.lt5` | 0.8394 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.equal.aids.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.equal.aids.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.rf.central.central.diabetic.single` | 0.4223 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.rf.central.central.diabetic.lt5` | 0.7800 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.xgb.central.central.diabetic.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.xgb.central.central.diabetic.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.rf.site.all.diabetic.single` | 0.4207 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.rf.site.all.diabetic.lt5` | 0.7674 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.all.diabetic.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.all.diabetic.lt5` | 0.0001 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.rf.site.dir0.1.diabetic.single` | 0.4258 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.rf.site.dir0.1.diabetic.lt5` | 0.7525 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.dir0.1.diabetic.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.dir0.1.diabetic.lt5` | 0.0004 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.rf.site.dir0.5.diabetic.single` | 0.4260 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.rf.site.dir0.5.diabetic.lt5` | 0.7649 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.dir0.5.diabetic.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.dir0.5.diabetic.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.rf.site.equal.diabetic.single` | 0.4145 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.rf.site.equal.diabetic.lt5` | 0.7761 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.equal.diabetic.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.equal.diabetic.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.rf.central.central.heart.single` | 0.4700 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.rf.central.central.heart.lt5` | 0.7444 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.xgb.central.central.heart.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.xgb.central.central.heart.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.rf.site.all.heart.single` | 0.4643 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.rf.site.all.heart.lt5` | 0.7429 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.all.heart.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.all.heart.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.rf.site.dir0.1.heart.single` | 0.4648 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.rf.site.dir0.1.heart.lt5` | 0.7446 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.dir0.1.heart.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.dir0.1.heart.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.rf.site.dir0.5.heart.single` | 0.4640 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.rf.site.dir0.5.heart.lt5` | 0.7473 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.dir0.5.heart.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.dir0.5.heart.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.rf.site.equal.heart.single` | 0.4642 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.rf.site.equal.heart.lt5` | 0.7392 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.equal.heart.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.default.leaf.xgb.site.equal.heart.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.rf.central.central.aids.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.rf.central.central.aids.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.central.central.aids.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.central.central.aids.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.all.aids.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.all.aids.lt5` | 0.0001 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.all.aids.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.all.aids.lt5` | 0.0001 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.dir0.1.aids.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.dir0.1.aids.lt5` | 0.0003 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.dir0.1.aids.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.dir0.1.aids.lt5` | 0.0002 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.dir0.5.aids.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.dir0.5.aids.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.dir0.5.aids.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.dir0.5.aids.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.equal.aids.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.equal.aids.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.equal.aids.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.equal.aids.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.rf.central.central.diabetic.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.rf.central.central.diabetic.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.central.central.diabetic.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.central.central.diabetic.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.all.diabetic.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.all.diabetic.lt5` | 0.0001 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.all.diabetic.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.all.diabetic.lt5` | 0.0001 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.dir0.1.diabetic.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.dir0.1.diabetic.lt5` | 0.0006 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.dir0.1.diabetic.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.dir0.1.diabetic.lt5` | 0.0003 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.dir0.5.diabetic.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.dir0.5.diabetic.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.dir0.5.diabetic.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.dir0.5.diabetic.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.equal.diabetic.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.equal.diabetic.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.equal.diabetic.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.equal.diabetic.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.rf.central.central.heart.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.rf.central.central.heart.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.central.central.heart.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.central.central.heart.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.all.heart.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.all.heart.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.all.heart.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.all.heart.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.dir0.1.heart.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.dir0.1.heart.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.dir0.1.heart.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.dir0.1.heart.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.dir0.5.heart.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.dir0.5.heart.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.dir0.5.heart.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.dir0.5.heart.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.equal.heart.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.rf.site.equal.heart.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.equal.heart.single` | 0.0000 |  |  | Single row leaf fraction, common draws. |
| `hp.regularized.leaf.xgb.site.equal.heart.lt5` | 0.0000 |  |  | Fewer than 5 distinct rows leaf fraction, common draws. |
| `anomaly.oob_fallback_equal.aids` | 0 |  |  | Count of site models. |
| `anomaly.oob_fallback_equal.diabetic` | 0 |  |  | Count of site models. |
| `anomaly.oob_fallback_equal.heart` | 0 |  |  | Count of site models. |
| `anomaly.oob_fallback_size.aids` | 0 |  |  | Count of site models. |
| `anomaly.oob_fallback_size.diabetic` | 0 |  |  | Count of site models. |
| `anomaly.oob_fallback_size.heart` | 0 |  |  | Count of site models. |
| `anomaly.oob_fallback_positives.aids` | 0 |  |  | Count of site models. |
| `anomaly.oob_fallback_positives.diabetic` | 0 |  |  | Count of site models. |
| `anomaly.oob_fallback_positives.heart` | 0 |  |  | Count of site models. |
| `anomaly.level2_a_nonpositive.aids` | 21596 |  |  | Count of level 2 fits (all evals and n_cal). |
| `anomaly.level2_a_nonpositive.diabetic` | 11000 |  |  | Count of level 2 fits (all evals and n_cal). |
| `anomaly.level2_a_nonpositive.heart` | 5664 |  |  | Count of level 2 fits (all evals and n_cal). |
| `anomaly.level2_a_nonpositive_central_L2.aids` | 1 |  |  | Count of level 2 fits, default view. |
| `anomaly.level2_a_nonpositive_fed_L2.aids` | 84 |  |  | Count of level 2 fits, default view. |
| `anomaly.level2_a_nonpositive_src_L2.aids` | 2906 |  |  | Count of level 2 fits, default view. |
| `anomaly.level2_a_nonpositive_central_L2.diabetic` | 0 |  |  | Count of level 2 fits, default view. |
| `anomaly.level2_a_nonpositive_fed_L2.diabetic` | 23 |  |  | Count of level 2 fits, default view. |
| `anomaly.level2_a_nonpositive_src_L2.diabetic` | 2186 |  |  | Count of level 2 fits, default view. |
| `anomaly.level2_a_nonpositive_central_L2.heart` | 0 |  |  | Count of level 2 fits, default view. |
| `anomaly.level2_a_nonpositive_fed_L2.heart` | 0 |  |  | Count of level 2 fits, default view. |
| `anomaly.level2_a_nonpositive_src_L2.heart` | 75 |  |  | Count of level 2 fits, default view. |
| `anomaly.single_class_at_0.5_central_L0.aids` | 0 |  |  | Count of metric rows, default view. |
| `anomaly.single_class_at_0.5_fed_L0.aids` | 79 |  |  | Count of metric rows, default view. |
| `anomaly.single_class_at_0.5_src_L0.aids` | 2414 |  |  | Count of metric rows, default view. |
| `anomaly.single_class_at_0.5_central_L0.diabetic` | 0 |  |  | Count of metric rows, default view. |
| `anomaly.single_class_at_0.5_fed_L0.diabetic` | 7 |  |  | Count of metric rows, default view. |
| `anomaly.single_class_at_0.5_src_L0.diabetic` | 1337 |  |  | Count of metric rows, default view. |
| `anomaly.single_class_at_0.5_central_L0.heart` | 0 |  |  | Count of metric rows, default view. |
| `anomaly.single_class_at_0.5_fed_L0.heart` | 307 |  |  | Count of metric rows, default view. |
| `anomaly.single_class_at_0.5_src_L0.heart` | 3134 |  |  | Count of metric rows, default view. |
| `anomaly.error_files.all` | 0 |  |  | Count of cells. |
