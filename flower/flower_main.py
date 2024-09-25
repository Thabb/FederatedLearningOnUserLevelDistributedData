import flwr as fl

from flower.client_loader import ClientLoader
from flower.evaluation import weighted_average
from flower.flwr.simulation import start_simulation

"""
This file contains all the different variables that might influence the outcome of training.
It also contains everything related to the simulation itself.
"""

# ======================================
# ===============SETTINGS===============
# ======================================

PROCESSING_UNIT: str = "cpu"  # "cpu" for cpu and "cuda" for GPU
NUM_CLIENTS: int = 500
BATCH_SIZE: int = 1
NUM_ROUNDS: int = 50
EPOCHS: int = 1
LEARNING_RATE: float = 0.001
IS_DAISY_CHAINING: bool = True  # False for Federated Averaging, True for Federated Daisy-Chaining
DAISY_CHAINING_CYCLE: int = 1  # Only used if IS_DAISY_CHAINING is "True", otherwise its ignored
AGGREGATE_CYCLE: int = 10  # Only used if IS_DAISY_CHAINING is "True", otherwise its ignored

# ========================================
# ===============SIMULATION===============
# ========================================

client_loader: ClientLoader = ClientLoader(NUM_CLIENTS, BATCH_SIZE, EPOCHS, PROCESSING_UNIT, LEARNING_RATE)

# Create strategy
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,  # Sample 100% of available clients for training
    fraction_evaluate=0.5,  # Sample 50% of available clients for evaluation
    min_fit_clients=NUM_CLIENTS,  # Never sample less than 10 clients for training
    min_evaluate_clients=int(NUM_CLIENTS / 2),  # Never sample less than 5 clients for evaluation
    min_available_clients=NUM_CLIENTS,  # Wait until all 10 clients are available
    evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
)


# Start simulation
start_simulation(
    client_fn=client_loader.client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources=client_loader.client_resources,
    is_daisy_chaining=IS_DAISY_CHAINING,
    daisy_chaining_cycle=DAISY_CHAINING_CYCLE,
    aggregate_cycle=AGGREGATE_CYCLE,
)
