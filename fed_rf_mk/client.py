import syft as sy
import numpy as np
import numpy.typing as npt
from typing import Union, TypeVar, TypedDict, Any, Dict, Tuple
import pandas as pd
from syft.service.policy.policy import MixedInputPolicy
from fed_rf_mk.utils import check_status_last_code_requests
import pickle
import cloudpickle
import random
import copy
import concurrent.futures

DataFrame = TypeVar("pandas.DataFrame")
NDArray = npt.NDArray[Any]
NDArrayInt = npt.NDArray[np.int_]
NDArrayFloat = npt.NDArray[np.float_]
Dataset = TypeVar("Dataset", bound=tuple[NDArrayFloat, NDArrayInt])

class DataParamsDict(TypedDict):
    target: str
    ignored_columns: list[Any]

class ModelParamsDict(TypedDict):
    model: bytes
    n_base_estimators: int
    n_incremental_estimators: int
    train_size: float
    test_size: float
    sample_size: int

DataParams = TypeVar("DataParams", bound=DataParamsDict)
ModelParams = TypeVar("ModelParams", bound=ModelParamsDict)


class FLClient:
    def __init__(self):
        self.datasites = {}
        self.eval_datasites = {}
        self.weights = {}
        self.dataParams = {}
        self.modelParams = {}
        self.model_parameters_history = {}
        self.shap_values_history = {}  # New: store SHAP values
        self.averaged_shap_values = None  # Store averaged SHAP values
        self.shap_arrays_fixed = False  # Track if SHAP arrays have been fixed

    
    def get_averaged_shap_values(self):
        """Return the averaged SHAP values from all silos."""
        if not self.shap_arrays_fixed:
            raise RuntimeError("SHAP arrays not generated. Please run start_shap_analysis() first.")

        return self.averaged_shap_values
    
    def add_train_client(self, name, url, email, password, weight = None):
        try:
            client = sy.login(email=email, password=password, url=url)
            self.datasites[name] = client
            self.weights[name] = weight
            print(f"Successfully connected to {name} at {url}")
        except Exception as e:
            print(f"Failed to connect to {name} at {url}: {e}")

    def add_eval_client(self, name, url, email, password):
        try:
            client = sy.login(email=email, password=password, url=url)
            self.eval_datasites[name] = client
            print(f"Successfully connected to {name} at {url}")
        except Exception as e:
            print(f"Failed to connect to {name} at {url}: {e}")
    
    def check_status(self):
        """
        Checks and prints the status of all connected silos.
        """
        for name, client in self.datasites.items():
            try:
                datasets = client.datasets
                print(f"{name}:  Connected ({len(datasets)} datasets available)")
            except Exception as e:
                print(f"{name}: Connection failed ({e})")

    def set_data_params(self, data_params):
        self.dataParams = data_params
        return f"Data parameters set: {data_params}"
    
    def set_model_params(self, model_params):
        self.modelParams = model_params
        return f"Model parameters set: {model_params}"

    def get_data_params(self):
        return self.dataParams

    def get_model_params(self):
        return self.modelParams

    def send_request(self):

        if not self.datasites:
            print("No clients connected. Please add clients first.")
            return
        
        if self.dataParams is None or self.modelParams is None:
            print("DataParams and ModelParams must be set before sending the request.")
            return
        
        for site in self.datasites:
            data_asset = self.datasites[site].datasets[0].assets[0]
            client = self.datasites[site]
            syft_fl_experiment = sy.syft_function(
                input_policy=MixedInputPolicy(
                    client=client,
                    data=data_asset,
                    dataParams=dict,
                    modelParams=dict
                )
            )(ml_experiment)
            ml_training_project = sy.Project(
                name="ML Experiment for FL",
                description="""Test project to run a ML experiment""",
                members=[client],
            )
            ml_training_project.create_code_request(syft_fl_experiment, client)
            project = ml_training_project.send()

        for site in self.eval_datasites:
            data_asset = self.eval_datasites[site].datasets[0].assets[0]
            client = self.eval_datasites[site]
            syft_fl_experiment = sy.syft_function(
                input_policy=MixedInputPolicy(
                    client=client,
                    data=data_asset,
                    dataParams=dict,
                    modelParams=dict
                )
            )(evaluate_global_model)
            ml_training_project = sy.Project(
                name="ML Evaluation for FL",
                description="""Test project to evaluate a ML model""",
                members=[client],
            )
            ml_training_project.create_code_request(syft_fl_experiment, client)
            project = ml_training_project.send()

    def check_status_last_code_requests(self):
        """
        Display status message of last code request sent to each datasite.
        """
        check_status_last_code_requests(self.datasites)
        check_status_last_code_requests(self.eval_datasites)


    def _average_shap_values(self, shap_dict, weights):
        """
        Average SHAP values across silos using weighted average.
        
        Args:
            shap_dict: Dict mapping silo_name -> shap_data
            weights: Dict mapping silo_name -> weight
            
        Returns:
            Dict with averaged SHAP values
        """
        if not shap_dict:
            return None
            
        # Get all feature names from first silo
        first_silo = next(iter(shap_dict.values()))
        all_features = set(first_silo['feature_names'])
        
        # Verify all silos have the same features
        for silo_name, shap_data in shap_dict.items():
            silo_features = set(shap_data['feature_names'])
            if silo_features != all_features:
                print(f"Warning: Feature mismatch in silo {silo_name}")
                all_features = all_features.intersection(silo_features)
        
        # Calculate weighted average for each feature's mean absolute SHAP value
        averaged_shap = {}
        for feature in all_features:
            weighted_sum = 0
            total_weight = 0
            
            for silo_name, shap_data in shap_dict.items():
                if feature in shap_data['mean_abs_shap'] and silo_name in weights:
                    weight = weights[silo_name]
                    weighted_sum += shap_data['mean_abs_shap'][feature] * weight
                    total_weight += weight
            
            if total_weight > 0:
                averaged_shap[feature] = weighted_sum / total_weight
        
        return {
            'feature_names': list(all_features),
            'mean_abs_shap': averaged_shap
        }
    
    # def print_shap_importances(self, top_n=10):
    #     """Print the top N most important features based on averaged SHAP values."""
    #     if not self.averaged_shap_values:
    #         print("No SHAP values available. Run the model first.")
    #         return
        
    #     # Sort by mean absolute SHAP value
    #     sorted_features = sorted(
    #         self.averaged_shap_values['mean_abs_shap'].items(),
    #         key=lambda x: x[1],
    #         reverse=True
    #     )
        
    #     print(f"\nTop {top_n} Most Important Features (Averaged SHAP values across silos):")
    #     print("-" * 70)
    #     for i, (feature, shap_value) in enumerate(sorted_features[:top_n], 1):
    #         print(f"{i:2d}. {feature:<40} {shap_value:.6f}")

    def get_shap_feature_importance_df(self):
        """Return SHAP importances as a pandas DataFrame."""
        if not self.shap_arrays_fixed:
            raise RuntimeError("SHAP arrays not generated. Please run start_shap_analysis() first.")

    
        if not self.averaged_shap_values:
            return None
            
        import pandas as pd
        return pd.DataFrame(
            list(self.averaged_shap_values['mean_abs_shap'].items()),
            columns=['Feature', 'Mean_Abs_SHAP']
        ).sort_values('Mean_Abs_SHAP', ascending=False)

    def get_silo_shap_values(self, silo_name):
        """
        Get SHAP values and importances for a specific silo.
        
        Args:
            silo_name (str): Name of the silo
            
        Returns:
            dict: SHAP data for the specified silo, or None if not found
        """
        if not self.shap_arrays_fixed:
            raise RuntimeError("SHAP arrays not generated. Please run start_shap_analysis() first.")


        if not self.shap_values_history:
            print("No SHAP values available. Run the model first.")
            return None
            
        if silo_name not in self.shap_values_history:
            print(f"Silo '{silo_name}' not found. Available silos: {list(self.shap_values_history.keys())}")
            return None
            
        silo_data = self.shap_values_history[silo_name].copy()
        silo_data['silo_weight'] = self.weights.get(silo_name, 0.0)

        df = pd.DataFrame(
            list(silo_data['mean_abs_shap'].items()),
            columns=['Feature', 'Mean_Abs_SHAP']
        ).sort_values('Mean_Abs_SHAP', ascending=False)
        
        return silo_data, df

    def get_weighted_average_shap_values(self):
        """
        Calculate weighted average of SHAP values across all silos using self.weights.
        
        Returns:
            dict: Weighted averaged SHAP values
        """
        if not self.shap_arrays_fixed:
            raise RuntimeError("SHAP arrays not generated. Please run start_shap_analysis() first.")


        if not self.shap_values_history:
            raise RuntimeError("No SHAP values available. Run the model first.")

        if not self.weights:
            raise RuntimeError("No weights available. Set weights first.")

        # Get all feature names from first silo
        first_silo_data = next(iter(self.shap_values_history.values()))
        all_features = set(first_silo_data['feature_names'])
        
        # Verify all silos have the same features and collect valid silos
        valid_silos = []
        for silo_name, shap_data in self.shap_values_history.items():
            if silo_name in self.weights:
                silo_features = set(shap_data['feature_names'])
                if silo_features != all_features:
                    print(f"Warning: Feature mismatch in silo {silo_name}")
                    all_features = all_features.intersection(silo_features)
                valid_silos.append(silo_name)
            else:
                print(f"Warning: No weight found for silo {silo_name}, excluding from average")
        
        if not valid_silos:
            print("No valid silos with weights found.")
            return None
        
        # Calculate weighted average for each feature's mean absolute SHAP value
        weighted_shap = {}
        total_samples = 0
        weights_used = {}
        
        # Normalize weights for valid silos only
        total_weight = sum(self.weights[silo] for silo in valid_silos)
        normalized_weights = {silo: self.weights[silo] / total_weight for silo in valid_silos}
        
        for feature in all_features:
            weighted_sum = 0
            
            for silo_name in valid_silos:
                shap_data = self.shap_values_history[silo_name]
                if feature in shap_data['mean_abs_shap']:
                    weight = normalized_weights[silo_name]
                    weighted_sum += shap_data['mean_abs_shap'][feature] * weight
                    weights_used[silo_name] = weight
            
            weighted_shap[feature] = weighted_sum
        
        # Calculate total samples (weighted sum)
        for silo_name in valid_silos:
            weight = normalized_weights[silo_name]
            sample_size = self.shap_values_history[silo_name]['sample_size']
            total_samples += sample_size * weight
        
        result = {
            'feature_names': list(all_features),
            'mean_abs_shap': weighted_shap,
            'total_samples': int(total_samples),
            'contributing_silos': valid_silos,
            'weights_used': weights_used
        }
        
        return result
    
    def run_model(self):
        modelParams = self.get_model_params()
        dataParams = self.get_data_params()

        all_estimators = []  # To store estimators from all silos in epoch 1
        modelParams_history = {}
        shap_values_history = {}  # New: store feature importances

        num_clients = len(self.weights)
        none_count = sum(1 for w in self.weights.values() if w is None)

        if none_count == num_clients:  
            # **Case 1: All weights are None → Assign equal weights**
            equal_weight = 1 / num_clients
            self.weights = {k: equal_weight for k in self.weights}
            print(f"All weights were None. Assigning equal weight: {equal_weight}")

        elif none_count > 0:
            # **Case 2: Some weights are None → Distribute remaining weight proportionally**
            defined_weights_sum = sum(w for w in self.weights.values() if w is not None)
            undefined_weight_share = (1 - defined_weights_sum) / none_count

            self.weights = {
                k: (undefined_weight_share if w is None else w) for k, w in self.weights.items()
            }
            print(f"Some weights were None. Distributing remaining weight: {self.weights}")

        # --- Federated loop ---
        for epoch in range(self.modelParams["fl_epochs"]):
            print(f"\nEpoch {epoch+1}/{self.modelParams['fl_epochs']}")

            if epoch == 0:
                # Parallel dispatch to all silos
                print("Launching first‐epoch training on all clients in parallel…")
                with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.datasites)) as executor:
                    # map future → client name
                    futures: dict[concurrent.futures.Future, str] = {}
                    for name, datasite in self.datasites.items():
                        data_asset = datasite.datasets[0].assets[0]
                        # push None as model for epoch 0
                        futures[executor.submit(
                            lambda ds, da, dp: ds.code.ml_experiment(
                                data=da, dataParams=dp, modelParams={**modelParams, "model": None}
                            ).get_from(ds),
                            datasite, data_asset, dataParams
                        )] = name

                    # collect results
                    for future in concurrent.futures.as_completed(futures):
                        name = futures[future]
                        try:
                            mp = future.result()
                            modelParams_history[name] = copy.deepcopy(mp)
                            
                            # NEW: shap values
                            if "shap_data" in mp:
                                shap_values_history[name] = mp["shap_data"]
                                print(f" ✔ {name} completed with shap analysis")
                            else:
                                print(f" ✔ {name} completed (no shap analysis)")
                                
                        except Exception as e:
                            print(f" ⚠️  {name} failed: {e}")

                # Renormalize weights to only the successful clients
                successful = list(modelParams_history.keys())
                total_w = sum(self.weights[n] for n in successful)
                self.weights = {n: self.weights[n] / total_w for n in successful}
                print(f"Re‐normalized weights among successful clients: {self.weights}")

                # NEW: Average SHAP values
                if shap_values_history:
                    self.averaged_shap_values = self._average_shap_values(
                        shap_values_history, self.weights
                    )
                    print("✔ SHAP values averaged across silos")

                # Merge their estimators
                print("Merging estimators from successful clients…")
                all_estimators = []
                merged_model = None
                for name, mp in modelParams_history.items():
                    clf = pickle.loads(mp["model"])
                    n_to_take = int(clf.n_estimators * self.weights[name])
                    all_estimators.extend(random.sample(clf.estimators_, n_to_take))
                    merged_model = clf

                # attach the merged ensemble
                merged_model.estimators_ = all_estimators
                modelParams["model"] = cloudpickle.dumps(merged_model)

            else:
                # subsequent epochs run sequentially on each client with the merged model
                for name, datasite in self.datasites.items():
                    data_asset = datasite.datasets[0].assets[0]
                    modelParams = datasite.code.ml_experiment(
                        data=data_asset,
                        dataParams=dataParams,
                        modelParams=modelParams
                    ).get_from(datasite)

        # Store histories
        self.model_parameters_history = modelParams_history
        self.shap_values_history = shap_values_history
        
        # store final merged modelParams
        self.set_model_params(modelParams)
        
    def run_evaluate(self):
        modelParams = self.get_model_params()
        dataParams = self.get_data_params()

        print(f"Number of evaluation sites: {len(self.eval_datasites)}")

        for name, datasite in self.eval_datasites.items():
            data_asset = datasite.datasets[0].assets[0]
            print(f"\nEvaluating model at {name}")

            # Send evaluation request
            model = datasite.code.evaluate_global_model(
                data=data_asset, dataParams=dataParams, modelParams=modelParams
            ).get_from(datasite)

            return model
        
    def start_shap_analysis(self):
        """Fix SHAP values that are currently stored as arrays."""
        print("Generating SHAP arrays...")
        
        # Fix individual silo data
        if self.shap_values_history:
            for silo_name, shap_data in self.shap_values_history.items():
                print(f"Generating SHAP arrays for silo: {silo_name}")
                fixed_mean_abs_shap = {}
                
                for feature, value in shap_data['mean_abs_shap'].items():
                    if isinstance(value, np.ndarray):
                        # Take the first element (they're identical anyway)
                        fixed_value = float(value[0])
                    else:
                        fixed_value = float(value)
                    fixed_mean_abs_shap[feature] = fixed_value
                
                # Update the silo data
                self.shap_values_history[silo_name]['mean_abs_shap'] = fixed_mean_abs_shap
        
        # Fix averaged data
        if self.averaged_shap_values and 'mean_abs_shap' in self.averaged_shap_values:
            print("Generating averaged SHAP values...")
            fixed_avg_shap = {}
            
            for feature, value in self.averaged_shap_values['mean_abs_shap'].items():
                if isinstance(value, np.ndarray):
                    fixed_value = float(value[0])
                else:
                    fixed_value = float(value)
                fixed_avg_shap[feature] = fixed_value
            
            # Update averaged data
            self.averaged_shap_values['mean_abs_shap'] = fixed_avg_shap

        print("✅ SHAP arrays generated!")
        self.shap_arrays_fixed = True



def evaluate_global_model(data: DataFrame, dataParams: dict, modelParams: dict) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix, matthews_corrcoef as mcc
    from sklearn.metrics import precision_score, recall_score, f1_score
    import pickle
    import numpy as np

    def preprocess(data: DataFrame) -> tuple[Dataset, Dataset]:

        # Step 1: Prepare the data for training
        # Drop rows with missing values in Q1
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
        _, X_test, _, y_test = train_test_split(X, y, test_size=modelParams["test_size"], stratify=y, random_state=42)
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
            "f1_score": f1_score(y_true, y_pred, average='weighted')
        }
    try:
        testing_data = preprocess(data)
        model = modelParams["model"]
        clf = pickle.loads(model)

        test_metrics = evaluate(clf, testing_data)

    except Exception as e:
        print(f"Error: {e}")
        test_metrics = {"error": str(e)}

    return test_metrics
    
def ml_experiment(data: DataFrame, dataParams: dict, modelParams: dict) -> dict:
    # preprocessing
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    import cloudpickle
    import pickle
    import numpy as np
    import sys
    from collections.abc import Mapping, Container

    def preprocess(data: DataFrame) -> tuple[Dataset, Dataset]:
        # Step 1: Prepare the data for training
        # Drop rows with missing values in Q1
        data = data.dropna(subset=[dataParams["target"]])

        # Separate features and target variable (Q1)
        y = data[dataParams["target"]]
        X = data.drop(dataParams["ignored_columns"], axis=1)

        # Step 2: Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, train_size=modelParams["train_size"], stratify=y, random_state=42
        )

        return (X_train, y_train), (X_test, y_test)

    def train(model, training_data: tuple[pd.DataFrame, pd.Series]) -> RandomForestClassifier:
        X_train, y_train = training_data
        model.fit(X_train, y_train)
        return model
    
    def evaluate(model, data: tuple[pd.DataFrame, pd.Series]) -> dict:
        X, y_true = data
        y_pred = model.predict(X)
        return {
            "accuracy": accuracy_score(y_true, y_pred)
        }
    
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
        
        if modelParams["model"]:
            model = modelParams["model"]
            clf = pickle.loads(model)
            clf.n_estimators += modelParams["n_incremental_estimators"]
        else:
            clf = RandomForestClassifier(
                random_state=42, 
                n_estimators=modelParams["n_base_estimators"], 
                warm_start=True
            )
        
        clf = train(clf, training_data)
        
        # NEW: Extract feature importances
        X_train, _ = training_data
        feature_names = X_train.columns.tolist()

        # Compute SHAP values using TreeExplainer (faster for tree models)
        import shap
        explainer = shap.TreeExplainer(clf)
        
        # Use a sample of training data for SHAP calculation (for performance)
        sample_size = min(100, len(X_train))
        X_sample = X_train.sample(n=sample_size, random_state=42)
        
        shap_values = explainer.shap_values(X_sample)
        
        # FIXED: Handle multi-class case properly
        if isinstance(shap_values, list):
            if len(shap_values) == 2:  # Binary classification
                # Use positive class (class 1)
                shap_vals = shap_values[1]
                print("DEBUG: Using positive class for binary classification")
            else:  # Multi-class
                # Average across classes
                shap_vals = np.mean(shap_values, axis=0)
                print(f"DEBUG: Averaging across {len(shap_values)} classes")
        else:
            # Single output (regression or single-class)
            shap_vals = shap_values
            print("DEBUG: Single output SHAP values")
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        shap_importance_dict = dict(zip(feature_names, mean_abs_shap))
        
        shap_data = {
            'shap_values': shap_vals,
            'feature_names': feature_names,
            'mean_abs_shap': shap_importance_dict,
            'sample_size': sample_size
        }        
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}

    # Prepare return dictionary
    result = {
        "model": cloudpickle.dumps(clf),
        "n_base_estimators": modelParams["n_base_estimators"],
        "n_incremental_estimators": modelParams["n_incremental_estimators"],
        "train_size": modelParams["train_size"],
        "sample_size": len(training_data[0]),
        "test_size": modelParams["test_size"],
        "shap_data": shap_data  # NEW: Add SHAP values instead of feature importances
    }

    # Calculate full size in bytes and megabytes
    size_in_bytes = deep_getsizeof(result, set())
    size_in_megabytes = size_in_bytes / (1024 * 1024)

    # Print the size
    print(f"Space occupied by result: {size_in_bytes} bytes ({size_in_megabytes:.2f} MB)")
    print(f"Number of features: {len(feature_names)}")
    print(f"SHAP sample size: {sample_size}")

    return result
