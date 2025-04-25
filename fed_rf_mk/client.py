import syft as sy
import numpy as np
import numpy.typing as npt
from typing import Union, TypeVar, Any, TypedDict, TypeVar
import pandas as pd
from syft.service.policy.policy import MixedInputPolicy
from utils import check_status_last_code_requests
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


    def run_model(self):
        modelParams = self.get_model_params()
        dataParams = self.get_data_params()

        all_estimators = []  # To store estimators from all silos in epoch 1
        modelParams_history = {}

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
                            print(f" ✔ {name} completed")
                        except Exception as e:
                            print(f" ⚠️  {name} failed: {e}")

                # Renormalize weights to only the successful clients
                successful = list(modelParams_history.keys())
                total_w = sum(self.weights[n] for n in successful)
                self.weights = {n: self.weights[n] / total_w for n in successful}
                print(f"Re‐normalized weights among successful clients: {self.weights}")

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
        # print("after predict")

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
        print(f"Testing data shape: {testing_data[0].shape}")
        model = modelParams["model"]
        clf = pickle.loads(model)
        print(f"Model estimators: {clf.n_estimators}")

        test_metrics = evaluate(clf, testing_data)
        print("CHEGOU AQUI")
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

    def preprocess(data: DataFrame) -> tuple[Dataset, Dataset]:

        # Step 1: Prepare the data for training
        # Drop rows with missing values in Q1
        data = data.dropna(subset=[dataParams["target"]])

        # Separate features and target variable (Q1)
        y = data[dataParams["target"]]
        X = data.drop(dataParams["ignored_columns"], axis=1)
        # # Replace inf/-inf with NaN, cast to float64, drop NaNs
        # X = X.replace([np.inf, -np.inf], np.nan).astype(np.float64)
        # mask = ~X.isnull().any(axis=1)
        # X = X[mask]
        # y = y[mask]


        # Step 2: Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=modelParams["train_size"], stratify=y, random_state=42)

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
    
    # Preprocess data
    try:
        training_data, test_data = preprocess(data)
        if modelParams["model"]:
            model = modelParams["model"]
            clf = pickle.loads(model)
            clf.n_estimators += modelParams["n_incremental_estimators"]
        else:
            clf = RandomForestClassifier(random_state=42, n_estimators=modelParams["n_base_estimators"], warm_start=True)
        clf = train(clf, training_data)
    except Exception as e:
        print(f"Error: {e}")
        return {"error": str(e)}
    
    # acc_train = evaluate(clf, training_data)["accuracy"]
    # acc_test = evaluate(clf, test_data)["accuracy"]
    # print(f"Train accuracy: {acc_train} - Test accuracy: {acc_test}: Difference: {acc_train - acc_test}")

    return {"model": cloudpickle.dumps(clf), "n_base_estimators": modelParams["n_base_estimators"], "n_incremental_estimators": modelParams["n_incremental_estimators"], "train_size": modelParams["train_size"], "sample_size": len(training_data[0]), "test_size": modelParams["test_size"]}

def hello_world():
    print("FedLearning RF is installed!")