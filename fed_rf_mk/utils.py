"""Collection of utility functions used throughout the ML experiments
to reduce boilerplate in the notebooks."""

import syft as sy
import joblib
import matplotlib.pyplot as plt

from IPython.display import display
from io import BytesIO
from pathlib import Path
from typing import Union
from sklearn.base import BaseEstimator

from itertools import product
from sklearn.metrics import ConfusionMatrixDisplay


def check_status_last_code_requests(datasites: dict[str, sy.DatasiteClient]) -> None:
    """display status message of last code request sent to each datasite"""
    for name, datasite in datasites.items():
        print(f"Datasite: {name}")
        display(datasite.code[-1].status)  # type: ignore

def approve_last_code_requests(datasites: dict[str, sy.DatasiteClient]) -> None:
    """Approve the last code request sent to each datasite"""
    for datasite in datasites.values():
        datasite.code[-1].status.approve()  # type: ignore

def requests_accepted(datasites: dict[str, sy.DatasiteClient]) -> list[bool]:
    """display status message of last code request sent to each datasite"""
    return [dsite.code[-1].status.approved for dsite in datasites.values()]

def plot_all_confusion_matrices(cms, title: str = None):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey="row")

    for coords, (name, cf_matrix) in zip(product(range(2), repeat=2), cms.items()):
        disp = ConfusionMatrixDisplay(cf_matrix, display_labels=["Absence", "Presence"])
        disp.plot(ax=axes[coords], xticks_rotation=45)
        disp.ax_.set_title(name)
        disp.im_.colorbar.remove()  # type: ignore
        disp.ax_.set_xlabel("")
        if coords[1] != 0:
            disp.ax_.set_ylabel("")

    fig.text(0.4, -0.05, "Predicted label", ha="left")
    plt.tight_layout()

    if title:
        plt.suptitle(title, y=1.05)

    fig.colorbar(disp.im_, ax=axes)
    return fig


def plot_fl_metrics(fl_metrics, title="Federated Learning Experiment Metrics"):
    """
    Plots the accuracy progression per epoch from federated learning metrics.

    Args:
        fl_metrics (list): A list of dictionaries containing metrics for each epoch.
                           Each dictionary should have an 'accuracy' key.
        title (str): The title of the plot.
    """
    plt.figure(figsize=(10, 6))

    # Extract accuracies from the metrics
    for question, metrics in fl_metrics.items():
        # print(f"Question: {question}, Metrics: {metrics}")
        epochs = list(range(1, len(metrics) + 1))
        accuracy = [m["accuracy"] for m in metrics]
        plt.plot(epochs, accuracy, marker='o', linestyle='-', label=question)

    plt.title(title, fontsize=14)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.xticks(epochs)  # Ensure all epochs are labeled
    plt.ylim(0, 1)  # Set Y-axis limits between 0 and 1
    # plt.yticks([i * 0.1 for i in range(11)])  # Set Y-axis ticks from 0 to 1 with a step of 0.1

    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()
