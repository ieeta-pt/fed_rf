"""Remote task functions executed on datasites.

These functions are sent to remote datasites via Syft code requests.
They intentionally import heavy dependencies inside function bodies to
reduce import-time side effects and to make remote execution self-contained.
"""

def evaluate_global_model(data, dataParams: dict, modelParams: dict) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        accuracy_score,
        confusion_matrix,
        matthews_corrcoef as mcc,
        precision_score,
        recall_score,
        f1_score,
        log_loss,
        roc_auc_score,
        average_precision_score,
    )
    import pickle
    import numpy as np
    import pandas as pd
    from sklearn.preprocessing import label_binarize
    from scipy.special import expit

    def preprocess(data) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
        # Step 1: Prepare the data for training
        data = data.dropna(subset=[dataParams["target"]])

        # Separate features and target variable (Q1)
        y = data[dataParams["target"]]
        X = data.drop(dataParams["ignored_columns"], axis=1)

        # Replace inf/-inf with NaN, cast to float64, drop NaNs
        X = X.replace([np.inf, -np.inf], np.nan).astype(np.float64)
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y = y[mask]

        # Step 2: Split the data into training and testing sets
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=modelParams["test_size"], stratify=y, random_state=42
        )
        return X_test, y_test

    def evaluate(model, data: tuple[pd.DataFrame, pd.Series]) -> dict:
        X, y_true = data
        X = X.replace([np.inf, -np.inf], np.nan).astype(np.float64)
        mask = ~X.isnull().any(axis=1)
        X = X[mask]
        y_true = y_true[mask]

        y_pred = model.predict(X)

        return {
            "mcc": mcc(y_true, y_pred),
            "cm": confusion_matrix(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted'),
            "f1_score": f1_score(y_true, y_pred, average='weighted'),
        }

    def _compute_metrics_from_preds(y_true, y_pred):
        return {
            "mcc": mcc(y_true, y_pred),
            "cm": confusion_matrix(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "precision": precision_score(y_true, y_pred, average='weighted'),
            "recall": recall_score(y_true, y_pred, average='weighted'),
            "f1_score": f1_score(y_true, y_pred, average='weighted'),
        }

    def _compute_probability_metrics(y_true, proba, classes):
        metrics = {}
        y_true_arr = np.asarray(y_true)
        classes_arr = np.asarray(classes)

        try:
            metrics["log_loss"] = float(log_loss(y_true_arr, proba, labels=classes_arr))
        except ValueError:
            metrics["log_loss"] = None

        y_indicator = label_binarize(y_true_arr, classes=classes_arr)
        if y_indicator.shape[1] == 1:
            y_indicator = np.hstack([1 - y_indicator, y_indicator])

        diff = y_indicator - proba
        metrics["brier_score"] = float(np.mean(np.sum(diff * diff, axis=1)))

        unique_labels = np.unique(y_true_arr)

        if proba.shape[1] == 2:
            if unique_labels.size > 1:
                try:
                    metrics["roc_auc"] = float(
                        roc_auc_score(y_true_arr, proba[:, 1], labels=classes_arr)
                    )
                except ValueError:
                    metrics["roc_auc"] = None

                try:
                    metrics["average_precision"] = float(
                        average_precision_score(
                            (y_true_arr == classes_arr[1]).astype(int),
                            proba[:, 1],
                        )
                    )
                except ValueError:
                    metrics["average_precision"] = None
            else:
                metrics["roc_auc"] = None
                metrics["average_precision"] = None
        else:
            if unique_labels.size > 1:
                try:
                    metrics["roc_auc"] = float(
                        roc_auc_score(
                            y_true_arr,
                            proba,
                            multi_class="ovr",
                            average="weighted",
                            labels=classes_arr,
                        )
                    )
                except ValueError:
                    metrics["roc_auc"] = None

                try:
                    metrics["average_precision"] = float(
                        average_precision_score(
                            y_indicator,
                            proba,
                            average="weighted",
                        )
                    )
                except ValueError:
                    metrics["average_precision"] = None
            else:
                metrics["roc_auc"] = None
                metrics["average_precision"] = None

        return metrics

    def _xgb_weighted_margin_proba(ensemble_members, X):
        """
        ensemble_members: list of (model_bytes, weight)
        Returns: (proba: np.ndarray [n_samples, n_classes], classes_out: np.ndarray [n_classes])
        """
        if not ensemble_members:
            raise ValueError("Empty XGB ensemble members.")

        import numpy as np
        import pickle

        # Normalize weights; fallback to uniform if sum <= 0
        ws = np.array([float(w) for _, w in ensemble_members], dtype=float)
        ws = ws / ws.sum() if ws.sum() > 0 else np.full(len(ws), 1.0 / max(len(ws), 1))
        print(f"Using ensemble weights: {ws}")
        
        margins_sum = None
        classes_ref = None
        n_classes = None

        for (model_bytes, w) in zip((m for m, _ in ensemble_members), ws):
            clf = pickle.loads(model_bytes)

            # First model defines reference class order
            if classes_ref is None:
                classes_ref = np.array(clf.classes_)
            # Pull margins from this model
            m = clf.predict(X, output_margin=True)
            m = np.asarray(m)

            # Binary: shape (n,) → treat as logit of "positive" (= classes_[1]).
            # Align positive class across members: if this model's positive class
            # differs from classes_ref[1], flip sign.
            if m.ndim == 1:
                n_classes = 2
                pos_label_this = clf.classes_[1]
                if pos_label_this == classes_ref[1]:
                    z = m  # already margin for ref positive class
                else:
                    # opposite ordering; flip sign so it refers to ref positive
                    z = -m
                z = z.reshape(-1, 1)  # keep as (n,1) aggregated margin for positive class

                margins_sum = (w * z) if margins_sum is None else (margins_sum + w * z)

            else:
                # Multi-class: shape (n, C). Align columns to classes_ref.
                n_classes = m.shape[1]
                # Build index map once
                if not np.array_equal(clf.classes_, classes_ref):
                    idx_map = {c: i for i, c in enumerate(clf.classes_)}
                    order = np.array([idx_map[c] for c in classes_ref])
                    m = m[:, order]
                margins_sum = (w * m) if margins_sum is None else (margins_sum + w * m)

        logits = margins_sum

        # Convert aggregated margins to probabilities
        if n_classes == 2 and margins_sum.shape[1] == 1:
            # Binary: have aggregated logit for positive class; derive probabilities
            z = margins_sum.ravel()
            p1 = expit(z)
            proba = np.column_stack([1 - p1, p1])
            logits = z.reshape(-1, 1)
        else:
            # Multi-class: softmax over aggregated logits
            m = margins_sum - margins_sum.max(axis=1, keepdims=True)
            e = np.exp(m)
            proba = e / e.sum(axis=1, keepdims=True)
            logits = margins_sum

        return proba, classes_ref, logits

    def _fit_binary_platt_logistic(logits_train, y_train, classes_ref):
        import numpy as np
        from sklearn.linear_model import LogisticRegression

        if logits_train is None or logits_train.size == 0:
            return None, {"status": "skipped", "reason": "empty_logits"}

        y_arr = np.asarray(y_train)
        unique = np.unique(y_arr)
        if unique.size < 2:
            return None, {"status": "skipped", "reason": "single_class_calibration"}

        # Map labels to {0,1} following classes_ref ordering
        pos_label = classes_ref[1]
        y_bin = (y_arr == pos_label).astype(int)

        clf = LogisticRegression(max_iter=200, solver="liblinear")
        clf.fit(logits_train.reshape(-1, 1), y_bin)

        class_index = {cls: idx for idx, cls in enumerate(clf.classes_)}
        if 0 not in class_index or 1 not in class_index:
            return None, {"status": "skipped", "reason": "logistic_missing_class"}

        def calibrate_fn(logits_eval):
            logits_eval = np.asarray(logits_eval).reshape(-1, 1)
            probs = clf.predict_proba(logits_eval)
            # Align to classes_ref ordering (negative, positive)
            negative = probs[:, class_index[0]].reshape(-1, 1)
            positive = probs[:, class_index[1]].reshape(-1, 1)
            return np.hstack([negative, positive])

        summary = {
            "status": "applied",
            "method": "platt_logistic",
            "coef": float(clf.coef_[0, 0]),
            "intercept": float(clf.intercept_[0]),
        }

        return calibrate_fn, summary

    def _fit_temperature_scaler(logits_train, y_train, classes_ref):
        import numpy as np
        from scipy.optimize import minimize_scalar
        from scipy.special import logsumexp

        if logits_train is None or logits_train.size == 0:
            return None, {"status": "skipped", "reason": "empty_logits"}

        logits_arr = np.asarray(logits_train)
        y_arr = np.asarray(y_train)
        class_to_idx = {cls: idx for idx, cls in enumerate(classes_ref)}

        try:
            y_idx = np.array([class_to_idx[cls] for cls in y_arr], dtype=int)
        except KeyError:
            return None, {"status": "skipped", "reason": "label_not_in_classes"}

        if logits_arr.ndim != 2 or logits_arr.shape[1] != len(classes_ref):
            return None, {"status": "skipped", "reason": "logits_shape_mismatch"}

        def nll(temp):
            temp = max(temp, 1e-3)
            scaled = logits_arr / temp
            log_probs = scaled - logsumexp(scaled, axis=1, keepdims=True)
            return -float(np.mean(log_probs[np.arange(logits_arr.shape[0]), y_idx]))

        result = minimize_scalar(nll, bounds=(0.05, 50.0), method="bounded")
        if not result.success:
            return None, {"status": "skipped", "reason": "opt_failure"}

        temperature = float(max(result.x, 1e-3))

        def calibrate_fn(logits_eval):
            logits_eval = np.asarray(logits_eval)
            scaled = logits_eval / temperature
            shifted = scaled - scaled.max(axis=1, keepdims=True)
            exps = np.exp(shifted)
            return exps / exps.sum(axis=1, keepdims=True)

        summary = {
            "status": "applied",
            "method": "temperature_scaling",
            "temperature": temperature,
            "opt_success": bool(result.success),
            "opt_fun": float(result.fun),
        }

        return calibrate_fn, summary

    try:
        X_test, y_test = preprocess(data)

        # XGBoost ensemble path (evaluation-time ensembling)
        if (
            modelParams.get("model_type") == "xgb"
            and modelParams.get("ensemble_members") is not None
            and len(modelParams.get("ensemble_members")) > 0
        ):
            # Ensure X_test is clean float64 (preprocess already did it)
            X_test = X_test.replace([np.inf, -np.inf], np.nan).astype(np.float64)
            mask = ~X_test.isnull().any(axis=1)
            X_eval = X_test[mask]
            y_eval = y_test[mask]

            proba_raw, classes = None, None
            logits = None
            try:
                proba_raw, classes, logits = _xgb_weighted_margin_proba(
                    modelParams["ensemble_members"], X_eval
                )
            except Exception as exc:  # pragma: no cover - defensive safety net
                raise RuntimeError(f"Failed to compute ensemble probabilities: {exc}") from exc

            calibration_cfg = modelParams.get("calibration", {}) or {}
            calibration_enabled = bool(calibration_cfg.get("enabled", True))
            calibration_fraction = float(calibration_cfg.get("fraction", 0.2))
            min_samples = int(calibration_cfg.get("min_samples", 25))
            binary_method = (calibration_cfg.get("binary_method") or "platt_logistic").lower()
            multiclass_method = (
                calibration_cfg.get("multiclass_method") or "temperature_scaling"
            ).lower()

            proba_eval = proba_raw
            y_eval_hold = y_eval
            calibration_summary = {
                "status": "skipped",
                "reason": "disabled" if not calibration_enabled else "not_requested",
            }

            n_eval = len(y_eval)
            is_binary = proba_raw.shape[1] == 2

            if (
                calibration_enabled
                and 0 < calibration_fraction < 1
                and n_eval >= max(min_samples, 4)
                and logits is not None
            ):
                idx_all = np.arange(n_eval)
                stratify = y_eval if isinstance(y_eval, pd.Series) else pd.Series(y_eval)
                try:
                    stratify_for_split = stratify if stratify.nunique() > 1 else None
                except Exception:
                    stratify_for_split = None

                try:
                    idx_cal, idx_hold = train_test_split(
                        idx_all,
                        train_size=calibration_fraction,
                        stratify=stratify_for_split,
                        random_state=42,
                    )
                except ValueError:
                    idx_cal, idx_hold = train_test_split(
                        idx_all,
                        train_size=calibration_fraction,
                        random_state=42,
                    )

                if idx_cal.size >= max(min_samples, 2) and idx_hold.size > 0:
                    if is_binary and binary_method == "platt_logistic":
                        calibrator, summary = _fit_binary_platt_logistic(
                            logits[idx_cal], y_eval.iloc[idx_cal], classes
                        )
                    elif (not is_binary) and multiclass_method == "temperature_scaling":
                        calibrator, summary = _fit_temperature_scaler(
                            logits[idx_cal], y_eval.iloc[idx_cal], classes
                        )
                    else:
                        calibrator, summary = None, {
                            "status": "skipped",
                            "reason": "unsupported_method",
                            "requested_binary_method": binary_method,
                            "requested_multiclass_method": multiclass_method,
                        }

                    if calibrator is not None:
                        proba_eval = calibrator(logits[idx_hold])
                        y_eval_hold = y_eval.iloc[idx_hold]
                        calibration_summary = {
                            **summary,
                            "calibration_fraction": float(calibration_fraction),
                            "calibration_size": int(idx_cal.size),
                            "holdout_size": int(idx_hold.size),
                        }
                    else:
                        calibration_summary = summary
                        proba_eval = proba_raw
                        y_eval_hold = y_eval
                else:
                    calibration_summary = {
                        "status": "skipped",
                        "reason": "insufficient_samples",
                        "calibration_fraction": float(calibration_fraction),
                        "n_eval": int(n_eval),
                    }

            y_pred = classes[np.argmax(proba_eval, axis=1)]
            test_metrics = _compute_metrics_from_preds(y_eval_hold, y_pred)
            test_metrics.update(_compute_probability_metrics(y_eval_hold, proba_eval, classes))
            test_metrics["calibration"] = calibration_summary

        elif modelParams.get("model_type") == "rf" and modelParams.get("model") is not None:
            # Fallback: single merged RF or single XGB model
            model = modelParams["model"]
            clf = pickle.loads(model)
            test_metrics = evaluate(clf, (X_test, y_test))
        else:
            raise ValueError("No valid model or ensemble members provided for evaluation.")

    except Exception as e:
        print("Evaluation error: %s", e)
        test_metrics = {"error": str(e)}

    return test_metrics


def ml_experiment(data, dataParams: dict, modelParams: dict) -> dict:
    # preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from sklearn.inspection import permutation_importance  # NEW: Import PFI
    import cloudpickle
    import pickle
    import numpy as np
    import sys
    from collections.abc import Mapping, Container
    import pandas as pd

    MODEL_RF = "rf"
    MODEL_XGB = "xgb"

    def preprocess(data) -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
        # Step 1: Prepare the data for training
        data = data.dropna(subset=[dataParams["target"]])

        # Separate features and target variable
        y = data[dataParams["target"]]
        X = data.drop(dataParams["ignored_columns"], axis=1)

        # # Align with evaluator: replace inf/-inf, coerce to float64, drop rows with NaNs
        # X = X.replace([np.inf, -np.inf], np.nan).astype(np.float64)
        # mask = ~X.isnull().any(axis=1)
        # X = X[mask]
        # y = y[mask]

        # Step 2: Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=modelParams["train_size"], stratify=y, random_state=42
        )

        return (X_train, y_train), (X_test, y_test)

    def train(model, training_data, xgb_prev_booster=None):
        X_train, y_train = training_data

        # If we're training XGBoost and continuing from a previous model, pass the booster
        if model_type == MODEL_XGB:
            # Choose an eval metric if the user didn't set one
            if model.get_xgb_params().get("eval_metric", None) is None:
                # binary -> logloss, multi -> mlogloss
                try:
                    n_classes = int(np.unique(y_train).shape[0])
                    eval_metric = "mlogloss" if n_classes > 2 else "logloss"
                except Exception:
                    eval_metric = "logloss"
                model.set_params(eval_metric=eval_metric)

            if xgb_prev_booster is not None:
                model.fit(X_train, y_train, xgb_model=xgb_prev_booster)
            else:
                model.fit(X_train, y_train)
        else:
            # Random Forest (existing behavior)
            model.fit(X_train, y_train)

        return model

    def evaluate(model, data):
        X, y_true = data
        y_pred = model.predict(X)
        return {"accuracy": accuracy_score(y_true, y_pred)}

    def deep_getsizeof(o, ids):
        """Recursively finds size of objects, including contents."""
        if id(o) in ids:
            return 0
        r = sys.getsizeof(o)
        ids.add(id(o))

        if isinstance(o, str) or isinstance(o, bytes):
            return r
        if isinstance(o, Mapping):
            return r + sum(deep_getsizeof(k, ids) + deep_getsizeof(v, ids) for k, v in o.items())
        if isinstance(o, Container):
            return r + sum(deep_getsizeof(i, ids) for i in o)
        return r

    # Preprocess data
    try:
        training_data, test_data = preprocess(data)

        # Decide which model to use
        model_type = modelParams.get("model_type", MODEL_RF)

        # --- Analysis consent and configuration ---
        # Request flag from orchestrator: True enables analysis (if silo policy allows)
        analysis_cfg = modelParams.get("analysis", {}) if isinstance(modelParams.get("analysis", {}), dict) else {}
        allow_analysis_req = bool(
            modelParams.get("allow_analysis", analysis_cfg.get("enabled", False))
        )
        # Silo-side policy switch (datasites module policy)
        from datasites import is_analysis_allowed as _site_policy_allows
        # Effective permission requires both the request and the silo policy to allow it
        do_analysis = allow_analysis_req and _site_policy_allows()

        # Fine-grained controls
        do_shap = bool(analysis_cfg.get("do_shap", True))
        do_pfi = bool(analysis_cfg.get("do_pfi", True))
        shap_sample_size = int(modelParams.get("shap_sample_size", analysis_cfg.get("shap_sample_size", 100)))
        pfi_n_repeats = int(modelParams.get("pfi_n_repeats", analysis_cfg.get("pfi_n_repeats", 5)))
        pfi_scoring = analysis_cfg.get("pfi_scoring", None)

        prev_booster = None

        if modelParams.get("model"):
            # Continuing from a previous serialized model
            clf = pickle.loads(modelParams["model"])

            if model_type == MODEL_RF:
                if not isinstance(clf, RandomForestClassifier):
                    raise TypeError("Loaded model is not a RandomForestClassifier.")
                clf.n_estimators += modelParams["n_incremental_estimators"]
                clf.warm_start = True

            elif model_type == MODEL_XGB:
                # Avoid importing xgboost here; check by name to prevent import errors on silos without xgboost
                if type(clf).__name__ != "XGBClassifier":
                    raise TypeError(f"Loaded model is not an XGBClassifier. Got: {type(clf)}")

                # Increase total number of trees and keep the old booster to continue training
                current_n = clf.get_params().get("n_estimators", 0)
                clf.set_params(n_estimators=current_n + modelParams["n_incremental_estimators"])
                try:
                    prev_booster = clf.get_booster()
                except Exception:
                    prev_booster = None
            else:
                raise ValueError(f"Unsupported model_type '{model_type}'. Use '{MODEL_RF}' or '{MODEL_XGB}'.")

        else:
            # Cold start
            if model_type == MODEL_RF:
                # Allow optional RF hyperparameters via modelParams; sensible defaults provided
                clf = RandomForestClassifier(
                    n_estimators=modelParams["n_base_estimators"],
                    random_state=42,
                    warm_start=True,
                    # criterion=modelParams.get("criterion", "gini"),
                    # max_depth=modelParams.get("max_depth", None),
                    # min_samples_split=modelParams.get("min_samples_split", 2),
                    # min_samples_leaf=modelParams.get("min_samples_leaf", 1),
                    # min_weight_fraction_leaf=modelParams.get("min_weight_fraction_leaf", 0.0),
                    # max_features=modelParams.get("max_features", "sqrt"),
                    # max_leaf_nodes=modelParams.get("max_leaf_nodes", None),
                    # min_impurity_decrease=modelParams.get("min_impurity_decrease", 0.0),
                    # bootstrap=modelParams.get("bootstrap", True),
                    # oob_score=modelParams.get("oob_score", False),
                    # n_jobs=modelParams.get("n_jobs", None),
                    # verbose=modelParams.get("verbose", 0),
                    # class_weight=modelParams.get("class_weight", None),
                    # ccp_alpha=modelParams.get("ccp_alpha", 0.0),
                    # max_samples=modelParams.get("max_samples", None),
                    # monotonic_cst=modelParams.get("monotonic_cst", None),
                )
            elif model_type == MODEL_XGB:
                # Allow optional XGB hyperparameters via modelParams; sensible defaults provided
                # Import XGBoost only when needed, and inside try block
                from xgboost import XGBClassifier  # gated import

                clf = XGBClassifier(
                    n_estimators=modelParams["n_base_estimators"],
                    random_state=42,
                    n_jobs=-1,
                    # Learning parameters
                    learning_rate=modelParams.get("learning_rate", 0.3),  # eta
                    max_depth=modelParams.get("max_depth", 6),
                    min_child_weight=modelParams.get("min_child_weight", 1),
                    gamma=modelParams.get("gamma", 0),  # min_split_loss
                    # Subsampling parameters
                    subsample=modelParams.get("subsample", 1.0),
                    colsample_bytree=modelParams.get("colsample_bytree", 1.0),
                    colsample_bylevel=modelParams.get("colsample_bylevel", 1.0),
                    colsample_bynode=modelParams.get("colsample_bynode", 1.0),
                    # Regularization parameters
                    reg_lambda=modelParams.get("reg_lambda", 1.0),  # lambda
                    reg_alpha=modelParams.get("reg_alpha", 0.0),   # alpha
                    # Tree construction parameters
                    tree_method=modelParams.get("tree_method", "auto"),
                    max_delta_step=modelParams.get("max_delta_step", 0),
                    scale_pos_weight=modelParams.get("scale_pos_weight", 1),
                    # System parameters
                    booster=modelParams.get("booster", "gbtree"),
                    device=modelParams.get("device", "cpu"),
                    verbosity=modelParams.get("verbosity", 1),
                    validate_parameters=modelParams.get("validate_parameters", True),
                    # Advanced tree parameters
                    grow_policy=modelParams.get("grow_policy", "depthwise"),
                    max_leaves=modelParams.get("max_leaves", 0),
                    max_bin=modelParams.get("max_bin", 256),
                    sampling_method=modelParams.get("sampling_method", "uniform"),
                )
            else:
                raise ValueError(
                    f"Unsupported model_type '{model_type}'. Use '{MODEL_RF}' or '{MODEL_XGB}'."
                )

        # Train (handles both RF and XGB; for XGB will continue if prev_booster is provided)
        clf = train(clf, training_data, xgb_prev_booster=prev_booster)

        shap_data = None
        pfi_data = None

        if do_analysis:
            # Extract feature importances using SHAP
            X_train, _ = training_data
            feature_names = X_train.columns.tolist()
            if do_shap:
                import shap  # gated import

                explainer = shap.TreeExplainer(clf)
                # Use a sample of training data for SHAP calculation (for performance)
                sample_size = min(shap_sample_size, len(X_train))
                X_sample = X_train.sample(n=sample_size, random_state=42)

                shap_values = explainer.shap_values(X_sample)

                # Handle multi-class case properly
                if isinstance(shap_values, list):
                    if len(shap_values) == 2:  # Binary classification
                        shap_vals = shap_values[1]  # positive class
                    else:  # Multi-class
                        shap_vals = np.mean(shap_values, axis=0)
                else:
                    shap_vals = shap_values

                # Calculate mean absolute SHAP values for each feature
                mean_abs_shap = np.abs(shap_vals).mean(axis=0)
                shap_importance_dict = dict(zip(feature_names, mean_abs_shap))

                shap_data = {
                    "shap_values": shap_vals,
                    "feature_names": feature_names,
                    "mean_abs_shap": shap_importance_dict,
                    "sample_size": sample_size,
                }

            if do_pfi:
                print("Calculating Permutation Feature Importance...")
                # Use the test set for PFI calculation to avoid overfitting
                X_test, y_test = test_data
                # Choose a more sensitive scoring when possible; honor override
                scoring_metric = pfi_scoring
                if scoring_metric is None:
                    scoring_metric = "accuracy"
                    try:
                        if hasattr(clf, "predict_proba"):
                            scoring_metric = "neg_log_loss"
                    except Exception:
                        scoring_metric = "accuracy"

                # Calculate PFI with multiple repetitions for robust estimates
                pfi_result = permutation_importance(
                    clf, X_test, y_test, n_repeats=pfi_n_repeats, random_state=42, scoring=scoring_metric
                )

                # Extract mean and std of importance scores
                pfi_mean_dict = dict(zip(feature_names, pfi_result.importances_mean))
                pfi_std_dict = dict(zip(feature_names, pfi_result.importances_std))

                pfi_data = {
                    "pfi_importances": pfi_result.importances,
                    "feature_names": feature_names,
                    "pfi_mean": pfi_mean_dict,
                    "pfi_std": pfi_std_dict,
                    "sample_size": len(X_test),
                    "n_repeats": pfi_n_repeats,
                    "scoring": scoring_metric,
                }

    except Exception as e:
        print("Training error: %s", e)
        return {"error": str(e)}

    # Prepare return dictionary
    try:
        result = {
            "model": cloudpickle.dumps(clf),
            "model_type": model_type,
            "n_base_estimators": modelParams["n_base_estimators"],
            "n_incremental_estimators": modelParams["n_incremental_estimators"],
            "train_size": modelParams["train_size"],
            "sample_size": len(training_data[0]),
            "test_size": modelParams["test_size"],
            # Analysis meta
            "analysis_status": (
                "enabled"
                if do_analysis
                else (
                    "disabled_by_silo_policy"
                    if allow_analysis_req and not _site_policy_allows()
                    else "disabled_by_request"
                )
            ),
        }

        # Include analysis outputs only when actually computed
        if shap_data is not None:
            result["shap_data"] = shap_data
        if pfi_data is not None:
            result["pfi_data"] = pfi_data

        # Calculate full size in bytes and megabytes
        size_in_bytes = deep_getsizeof(result, set())
        size_in_megabytes = size_in_bytes / (1024 * 1024)

        print(
            "Space occupied by result: %s bytes (%.2f MB)", size_in_bytes, size_in_megabytes
        )

        return result
    except Exception as e:
        print("Result packaging error: %s", e)
        return {"error": str(e)}
