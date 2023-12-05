import torch

import flwr as fl

from flower.data_loader import load_datasets
from flower.evaluation import weighted_average
from model import Net
from client import FlowerClient
"""
This file contains all the different variables that might influence the outcome of training.
It also contains everything related to the simulation itself.
"""

# ======================================
# ===============SETTINGS===============
# ======================================

DEVICE = torch.device("cpu")  # "cpu" for cpu and "cuda" for GPU

# Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
client_resources = None
if DEVICE.type == "cuda":
    client_resources = {"num_gpus": 1, "num_cpus": 1}

print(
    f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
)

NUM_CLIENTS = 10
BATCH_SIZE = 1
NUM_ROUNDS = 10
EPOCHS = 1


# // TODO: Place this somewhere else, so that this file contains just the simulation settings?
CLASSES = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# TODO: Move this line and the function below into the client.py file
train_loaders, val_loaders, test_loader = load_datasets(NUM_CLIENTS, BATCH_SIZE)


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net().to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    train_loader = train_loaders[int(cid)]
    val_loader = val_loaders[int(cid)]

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, train_loader, val_loader, EPOCHS)


# ========================================
# ===============SIMULATION===============
# ========================================

# Create FedAvg strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=int(NUM_CLIENTS / 2),  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
)

# Start simulation
fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources=client_resources,
)
