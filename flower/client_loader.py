import torch
import flwr as fl

from torch import device
from torch.utils.data import DataLoader

from flower.client import FlowerClient
from flower.data_loader import load_datasets
from model import Net


class ClientLoader:
    """
    Loads a client and loads the dataset depending on the number of clients and the batch size.
    """
    train_loaders: [DataLoader] = []
    val_loaders: [DataLoader] = []
    test_loaders: DataLoader = None
    EPOCHS: int = 1
    DEVICE: device = None
    client_resources: dict = None

    def __init__(self, num_clients: int, batch_size: int, epochs: int, processing_unit: str):
        """
        Initializes the different dataloaders and splits them between the number of clients.

        :param num_clients: Number of clients used in the simulation.
        :param batch_size: Number of examples that are used in one batch.
        :param epochs: Number of times the examples are looked at in one round.
        :param processing_unit: Is the simulation done on a "cpu" or "cuda" (GPU).
        """
        self.train_loaders, self.val_loaders, self.test_loader = load_datasets(num_clients, batch_size)
        self.EPOCHS = epochs
        self.DEVICE = torch.device(processing_unit)

        # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
        # this strongly depends on the physical device, that is being used to run the simulation
        if self.DEVICE.type == "cuda":
            self.client_resources = {"num_gpus": 0.1, "num_cpus": 1}

        # helps a lot to notice the wrong device/version being used
        print(
            f"Training on {self.DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
        )

    def client_fn(self, cid: str) -> FlowerClient:
        """
        Create a Flower client representing a single organization.
        This function will be called during the simulation process.

        :param cid: Id for the client.
        :return: A fully functioning client that is ready to be used in the simulation.
        """

        # Load model
        net = Net().to(self.DEVICE)

        # Load data
        # Note: each client gets a different trainloader/valloader, so each client
        # will train and evaluate on their own unique data
        train_loader = self.train_loaders[int(cid)]
        val_loader = self.val_loaders[int(cid)]

        # Create a  single Flower client representing a single organization
        return FlowerClient(net, train_loader, val_loader, self.EPOCHS)
