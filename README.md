# fed_rf_mk

A Python package for implementing federated learning with Random Forests using [PySyft](https://github.com/OpenMined/PySyft).

## Description

`fed_rf_mk` is a federated learning implementation that allows multiple parties to collaboratively train Random Forest models without sharing their raw data. This package leverages PySyft's secure federated learning framework to protect data privacy while enabling distributed model training.

Key features:
- Secure federated training of Random Forest classifiers
- Weighted model averaging based on client importance
- Incremental learning approach for multi-round training
- Evaluation of global models on local test data
- Support for both training and evaluation clients

## Installation

### Prerequisites

- Python 3.10.12 or higher

### Installing from PyPI

```bash
pip install fed_rf_mk
```

### Installing from Source

```bash
git clone https://github.com/ieeta-pt/fed_rf.git
cd fed_rf
pip install -e .
```

## Setting Up a Federated Learning Environment

### 1. Launch Data Silos (Servers)

There are multiple ways to launch PySyft servers, each representing a data silo with its local dataset:


#### Option 1: Create a custom launcher script

Create a `main.py` file:

```python
import argparse
from fed_rf_mk.server import launch_datasite

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch a single DataSite server independently.")

    parser.add_argument("--name", type=str, required=True, help="The name of the DataSite (e.g., silo1, silo2, etc.)")
    parser.add_argument("--port", type=int, required=True, help="The port number for the DataSite")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the dataset")
    parser.add_argument("--mock_path", type=str, help="Path to mock dataset")

    args = parser.parse_args()

    launch_datasite(name=args.name, port=args.port, data_path=args.data_path, mock_path=args.mock_path)
```

Then run it:

```bash
python main.py --name silo1 --port 8080 --data_path path/to/data1.csv
```

#### Option 2: Launch programmatically in your code

```python
from fed_rf_mk.server import launch_datasite

# Launch a server directly from your code
launch_datasite(name="silo1", port=8080, data_path="path/to/data1.csv", mock_path="path/to/mock.csv")
```

### 2. Set Up the Federated Learning Client

With the servers running, you can now set up a federated learning client:

```python
from fed_rf_mk.client import FLClient

# Initialize the client
fl_client = FLClient()

# Add training clients with weights (weight parameter is optional)
fl_client.add_train_client(
    name="silo1", 
    url="http://localhost:8080", 
    email="fedlearning@rf.com", 
    password="****", 
    weight=0.4
)
fl_client.add_train_client(
    name="silo2", 
    url="http://localhost:8081", 
    email="fedlearning@rf.com", 
    password="****"
)

# Add evaluation client (doesn't contribute to training but evaluates the model)
fl_client.add_eval_client(
    name="silo3", 
    url="http://localhost:8082", 
    email="fedlearning@rf.com", 
    password="****"
)
```

### 3. Configure Data and Model Parameters

Define the parameters for your data preprocessing and Random Forest model:

```python
# Define data parameters
data_params = {
    "target": "target_column",              # Target column name
    "ignored_columns": ["id", "timestamp"]  # Columns to exclude from training
}

# Define model parameters
model_params = {
    "model": None,                  # Initial model (None for first round)
    "n_base_estimators": 100,       # Number of trees for the initial model
    "n_incremental_estimators": 10, # Number of trees to add in each subsequent round
    "train_size": 0.8,              # Proportion of data to use for training
    "test_size": 0.2,               # Proportion of data to use for testing
    "sample_size": None,            # Sample size for training (None uses all available data)
    "fl_epochs": 3                  # Number of federated learning rounds
}

# Set parameters in the client
fl_client.set_data_params(data_params)
fl_client.set_model_params(model_params)
```

### 4. Send Requests to Clients

Send the code execution requests to all clients:

```python
fl_client.send_request()

# Check the status of the requests
fl_client.check_status_last_code_requests()
```

### 5. Run the Federated Training

Execute the federated learning process:

```python
fl_client.run_model()
```

This will:
1. Train local models on each client
2. Collect and aggregate the models based on weights
3. Run multiple federated rounds (controlled by the `fl_epochs` parameter in modelParams)

### 6. Evaluate the Federated Model

Finally, evaluate the federated model on the evaluation clients:

```python
evaluation_results = fl_client.run_evaluate()
print(evaluation_results)
```

## Complete Example

Below is a complete example workflow based on the provided `main.ipynb`:

```python
from fed_rf_mk.client import FLClient

# Initialize the client
rf_client = FLClient()

# Connect to data silos
rf_client.add_train_client(name="silo1", url="http://localhost:8080", email="fedlearning@rf.com", password="****", weight=0.4)
rf_client.add_train_client(name="silo2", url="http://localhost:8081", email="fedlearning@rf.com", password="****")
rf_client.add_eval_client(name="silo3", url="http://localhost:8082", email="fedlearning@rf.com", password="****")

# Define parameters
questions = ['Q' + str(i) for i in range(1, 13)]
dataParams = {
    "target": "Q1",
    "ignored_columns": ["patient_id", "source"] + questions
}

modelParams = {
    "model": None,
    "n_base_estimators": 100,
    "n_incremental_estimators": 1,
    "train_size": 0.2,
    "test_size": 0.5,
    "sample_size": None,
    "fl_epochs": 2
}

rf_client.set_data_params(dataParams)
rf_client.set_model_params(modelParams)

# Send requests
rf_client.send_request()
rf_client.check_status_last_code_requests()

# Run federated training
rf_client.run_model()

# Evaluate model
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

## Understanding the Code Architecture

The package is organized as follows:

- `client.py`: Contains the main `FLClient` class for orchestrating federated learning
- `server.py`: Provides functions for launching and managing PySyft servers
- `datasites.py`: Handles dataset creation and server setup
- `datasets.py`: Contains utilities for data processing
- `utils.py`: Provides helper functions for visualization and communication

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- [PySyft](https://github.com/OpenMined/PySyft) for the secure federated learning framework
- [scikit-learn](https://scikit-learn.org/) for the Random Forest implementation
