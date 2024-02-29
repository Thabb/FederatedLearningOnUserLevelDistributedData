import flwr as fl

from flower.client_loader import ClientLoader
from flower.evaluation import weighted_average

"""
This file contains all the different variables that might influence the outcome of training.
It also contains everything related to the simulation itself.
"""

# ======================================
# ===============SETTINGS===============
# ======================================

PROCESSING_UNIT: str = "cpu"  # "cpu" for cpu and "cuda" for GPU
NUM_CLIENTS: int = 10
BATCH_SIZE: int = 1
NUM_ROUNDS: int = 5
EPOCHS: int = 1

# ========================================
# ===============SIMULATION===============
# ========================================

client_loader: ClientLoader = ClientLoader(NUM_CLIENTS, BATCH_SIZE, EPOCHS, PROCESSING_UNIT)

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
    client_fn=client_loader.client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources=client_loader.client_resources,
)
