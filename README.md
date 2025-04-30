# fed_rf_mk

[![PyPI version](https://badge.fury.io/py/fed-rf-mk.svg)](https://badge.fury.io/py/fed-rf-mk)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A Python package for implementing federated learning with Random Forests using [PySyft](https://github.com/OpenMined/PySyft). This implementation allows multiple parties to collaboratively train Random Forest models without sharing their raw data, maintaining privacy and confidentiality while leveraging the combined knowledge of all participants.

## Features

- Secure federated training of Random Forest classifiers using PySyft's framework
- Client-server architecture for federated learning
- Weighted model averaging based on client importance
- Incremental learning approach for multi-round training
- Evaluation of global models on local test data
- Support for both training and evaluation clients
- Configurable model parameters
- Simple API for both client and server implementations

## Installation

### Prerequisites

- Python 3.10.12 or higher

### Installing from PyPI

```bash
pip install fed-rf-mk
```

### Installing from Source

```bash
git clone https://github.com/ieeta-pt/fed_rf.git
cd fed_rf
pip install -e .
```

## Federated Learning Workflow

The complete federated learning process follows these steps:

### 1. Launch Data Silos (Servers)

First, start all server instances that will participate in the federated learning process:

```python
from fed_rf_mk.server import FLServer
import threading

# Start the first server
server1 = FLServer("silo1", 8080, "path/to/data1.csv", auto_accept=False)
server_thread1 = threading.Thread(target=server1.start, daemon=True)
server_thread1.start()

# Start the second server
server2 = FLServer("silo2", 8081, "path/to/data2.csv", auto_accept=False)
server_thread2 = threading.Thread(target=server2.start, daemon=True)
server_thread2.start()

# Start the evaluation server
server3 = FLServer("eval_silo", 8082, "path/to/eval_data.csv", auto_accept=False)
server_thread3 = threading.Thread(target=server3.start, daemon=True)
server_thread3.start()
```

### 2. Set Up the Federated Learning Client

Next, initialize the client and connect to all participating servers:

```python
from fed_rf_mk.client import FLClient

# Initialize client
fl_client = FLClient()

# Add training clients
fl_client.add_train_client(
    name="silo1",
    url="http://localhost:8080", 
    email="fedlearning@rf.com", 
    password="your_password"
)
fl_client.add_train_client(
    name="silo2",
    url="http://localhost:8081", 
    email="fedlearning@rf.com", 
    password="your_password"
)

# Add evaluation client
fl_client.add_eval_client(
    name="eval_silo",
    url="http://localhost:8082", 
    email="fedlearning@rf.com", 
    password="your_password"
)
```

### 3. Configure Data and Model Parameters

Define the parameters for your data and the Random Forest model:

```python
# Define data parameters
data_params = {
    "target": "target_column",              # Target column name
    "ignored_columns": ["id", "timestamp"]  # Columns to exclude from training
}
fl_client.set_data_params(data_params)

# Define model parameters
model_params = {
    "model": None,                  # Initial model (None for first round)
    "n_base_estimators": 100,       # Number of trees for the initial model
    "n_incremental_estimators": 10, # Number of trees to add in each round
    "train_size": 0.8,              # Proportion of data for training
    "test_size": 0.2,               # Proportion of data for testing
    "sample_size": None,            # Sample size (None uses all data)
    "fl_epochs": 3                  # Number of federated learning rounds
}
fl_client.set_model_params(model_params)
```

### 4. Send Requests to Servers

Send the federated learning request to all participating servers:

```python
# Send the request
fl_client.send_request()

# Check the status of sent requests
fl_client.check_status_last_code_requests()
```

### 5. Approve Requests on Servers

Each server needs to review and approve the federated learning request before training can begin:

```python
# On server1
server1.list_pending_requests()  # See all pending requests
server1.inspect_request(0)       # Examine request details (code, parameters)
server1.approve_request(0)       # Approve request #0

# On server2
server2.approve_request(0)

# On server3 (evaluation server)
server3.approve_request(0)
```

### 6. Run Federated Training

After all servers have approved the request, start the federated training process:

```python
# Start federated training
fl_client.run_model()
```

This will:
1. Train local models on each client's data
2. Aggregate the models using weighted averaging
3. Run for multiple epochs if specified in model parameters

### 7. Evaluate the Federated Model

Finally, evaluate the performance of your federated model on the evaluation data:

```python
# Run evaluation
evaluation_results = fl_client.run_evaluate()
print(evaluation_results)
```

## Example Use Case: Clinical Trials Analysis

The example below demonstrates how to use federated random forests for analyzing clinical trial data where data is distributed across multiple sites:

```python
# 1. Distribute data across sites (based on the attached example notebook)
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.utils import shuffle

# Fetch dataset
aids_clinical_trials_group_study_175 = fetch_ucirepo(id=890)

# Extract data
X = aids_clinical_trials_group_study_175.data.features
y = aids_clinical_trials_group_study_175.data.targets
df = pd.concat([X, y], axis=1)

# Separate classes for balanced distribution
df_majority = df[df["cid"] == 0]  # cid = 0 (majority)
df_minority = df[df["cid"] == 1]  # cid = 1 (minority)

# Shuffle and split data
df_majority = shuffle(df_majority, random_state=42).reset_index(drop=True)
df_minority = shuffle(df_minority, random_state=42).reset_index(drop=True)

# Create partitions
N = 3  # Number of partitions
TRAIN_RATIO = 0.8  # 80% training, 20% testing

# Split data into partitions and save
# (Partition code omitted for brevity - see notebook for details)

# 2. Launch servers for each data partition
from fed_rf_mk.server import FLServer
import threading

# Server for partition 0
server1 = FLServer("aids_clinical_part_0", 8080, "train_datasets/aids_clinical/part_0.csv", auto_accept=False)
server_thread1 = threading.Thread(target=server1.start, daemon=True)
server_thread1.start()

# Server for partition 1
server2 = FLServer("aids_clinical_part_1", 8081, "train_datasets/aids_clinical/part_1.csv", auto_accept=False)
server_thread2 = threading.Thread(target=server2.start, daemon=True)
server_thread2.start()

# Server for test partition (evaluation)
server3 = FLServer("aids_clinical_part_2", 8082, "train_datasets/aids_clinical/part_2.csv", auto_accept=False)
server_thread3 = threading.Thread(target=server3.start, daemon=True)
server_thread3.start()

# 3. Set up client to coordinate federated learning
from fed_rf_mk.client import FLClient

rf_client = FLClient()

# Add training clients
rf_client.add_train_client(name="aids_clinical_part_0", url="http://localhost:8080", 
                          email="fedlearning@rf.com", password="your_password")
rf_client.add_train_client(name="aids_clinical_part_1", url="http://localhost:8081", 
                          email="fedlearning@rf.com", password="your_password")

# Add evaluation client
rf_client.add_eval_client(name="aids_clinical_part_2", url="http://localhost:8082", 
                         email="fedlearning@rf.com", password="your_password")

# 4. Configure learning parameters
data_params = {
    "target": "cid",
    "ignored_columns": ["cid"]
}

model_params = {
    "model": None,
    "n_base_estimators": 10,
    "n_incremental_estimators": 2,
    "train_size": 0.7,
    "test_size": 0.5,
    "sample_size": None,
    "fl_epochs": 1
}

rf_client.set_data_params(data_params)
rf_client.set_model_params(model_params)

# 5. Send request to all servers
rf_client.send_request()

# 6. Approve requests on servers
server1.list_pending_requests()  # Check pending requests
server1.inspect_request(0)       # Inspect request details
server1.approve_request(0)       # Approve the request

server2.approve_request(0)
server3.approve_request(0)

# 7. Train federated model
rf_client.run_model()

# 8. Evaluate model
evaluation_results = rf_client.run_evaluate()
print(evaluation_results)
```

## Client Weighting

The package supports weighted aggregation of models based on client importance. You can:

1. **Explicitly assign weights**: Provide a weight for each client when adding them:
   ```python
   fl_client.add_train_client(name="silo1", url="url", email="email", password="pwd", weight=0.6)
   fl_client.add_train_client(name="silo2", url="url", email="email", password="pwd", weight=0.4)
   ```

2. **Mixed weighting**: Assign weights to some clients and let others be calculated automatically:
   ```python
   fl_client.add_train_client(name="silo1", url="url", email="email", password="pwd", weight=0.6)
   fl_client.add_train_client(name="silo2", url="url", email="email", password="pwd") # Weight will be calculated
   ```

3. **Equal weighting**: Don't specify any weights, and all clients will receive equal weight.

## API Reference

### FLServer

The `FLServer` class represents a data provider in the federated learning system.

```python
FLServer(
    name: str,
    port: int,
    data_path: str,
    auto_accept: bool = False
)
```

**Parameters:**
- `name`: Unique identifier for the server
- `port`: Port to host the server on
- `data_path`: Path to the CSV file with training data
- `auto_accept`: If True, automatically accepts federated learning requests

**Methods:**
- `start()`: Start the server
- `list_pending_requests()`: List all pending federated learning requests
- `inspect_request(request_id)`: View details of a specific request
- `approve_request(request_id)`: Approve a federated learning request
- `reject_request(request_id)`: Reject a federated learning request

### FLClient

The `FLClient` class coordinates the federated learning process across multiple data providers.

```python
FLClient()
```

**Methods:**
- `add_train_client(name, url, email, password, weight=None)`: Add a training data source with optional weight
- `add_eval_client(name, url, email, password)`: Add an evaluation data source
- `set_data_params(params)`: Configure data parameters
- `set_model_params(params)`: Configure model parameters
- `send_request()`: Send federated learning requests to all data sources
- `check_status_last_code_requests()`: Check status of all pending requests
- `run_model()`: Train the federated model
- `get_model_params()`: Get the parameters of the trained model
- `run_evaluate()`: Evaluate the model on the evaluation data source and return results

**Data Parameters:**
- `target`: Target variable column name
- `ignored_columns`: List of column names to exclude from training

**Model Parameters:**
- `model`: Pre-trained model (None to create new)
- `n_base_estimators`: Number of base estimators for the Random Forest
- `n_incremental_estimators`: Number of new estimators to add per epoch
- `train_size`: Fraction of data to use for training
- `test_size`: Fraction of data to use for testing
- `sample_size`: Number of samples to use (None for all)
- `fl_epochs`: Number of federated learning epochs

## Data Format

The library expects data in CSV format with the following requirements:
- All servers should have compatible data schemas (same column names and types)
- The target variable should be present in all data files
- Categorical variables should be properly encoded before use

## Understanding the Code Architecture

The package is organized as follows:

- `client.py`: Contains the main `FLClient` class for orchestrating federated learning
- `server.py`: Provides the `FLServer` class for hosting data and processing requests
- `datasets.py`: Contains utilities for data processing
- `utils.py`: Provides helper functions for visualization and communication

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## How to cite

If you use **fed_rf_mk** in your work, please cite:

```bibtex
@inproceedings{cotorobai2025federated,
  title     = {A Federated Random Forest Solution for Secure Distributed Machine Learning},
  author    = {Cotorobai, Alexandre and Silva, Jorge Miguel and Oliveira, Jos{\'e} Luis},
  booktitle = {Proceedings of the IEEE International Conference on Bioinformatics and Biomedicine (CBMS)},
  year      = {2025},
  pages     = {–},      % update when available
  doi       = {–}       % update when available
}

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [PySyft](https://github.com/OpenMined/PySyft) for the secure federated learning framework
- [scikit-learn](https://scikit-learn.org/) for the Random Forest implementation
