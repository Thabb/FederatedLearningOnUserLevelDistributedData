from collections import OrderedDict
from typing import List, Tuple, Dict

import numpy as np
import torch
import flwr as fl
from torch import device
from flwr.common import Scalar
from torch.nn import Module
from torch.utils.data import DataLoader


class FlowerClient(fl.client.NumPyClient):
    """
    Representation of a single client. Multiple clients are initialized during the simulation.
    Clients hold all the logic needed for training and evaluation.
    """

    def __init__(self, net: Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int,
                 device: device, learning_rate: float):
        """
        Initializes the client with all data that is needed for training and evaluation of the model.
        :param net: Neural network.
        :param train_loader: Dataloaders containing the training data.
        :param val_loader: Dataloaders containing the evaluation data.
        :param epochs: Number of times the data is looked at.
        """
        self.net: Module = net
        self.train_loader: DataLoader = train_loader
        self.val_loader: DataLoader = val_loader
        self.epochs: int = epochs
        self.device: device = device
        self.learning_rate = learning_rate

    def get_parameters(self, config: Dict[str, Scalar]) -> List[np.ndarray]:
        """
        Gets the parameters of a NN.
        :param config: Config file only used in the same method inside the parent class.
        :return: List of numpy arrays containing the parameters.
        """
        return [val.cpu().numpy() for _, val in self.net.state_dict().items()]

    def fit(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[List[np.ndarray], int, dict]:
        """
        Train the provided parameters using the locally held dataset.
        :param parameters: List of numpy arrays containing the parameters.
        :param config: Config file used in the same method inside the parent class.
        :return: A list of new parameters, combined with the amount of training data.
        """
        self.set_parameters(self.net, parameters)
        self.train(self.net, self.train_loader, epochs=self.epochs)
        return self.get_parameters({}), len(self.train_loader), {}

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, Scalar]) -> Tuple[float, int, dict]:
        """
        Evaluate the provided parameters using the locally held dataset
        :param parameters: List of numpy arrays containing the parameters.
        :param config: Config file used in the same method inside the parent class.
        :return: The calculated loss, amount of evaluation data and a dict containing the calculated accuracy.
        """
        self.set_parameters(self.net, parameters)
        loss, accuracy = self.test(self.net, self.val_loader)
        return float(loss), len(self.val_loader), {"accuracy": float(accuracy)}

    def train(self, net: Module, train_loader: DataLoader, epochs: int, verbose: bool = False) -> None:
        """
        Trains the network on the training set.
        :param net: Neural Network.
        :param train_loader: Data loader containing the training data.
        :param epochs: Number of times the training data should be iterated.
        :param verbose: Should there be a print of information?
        """
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=self.learning_rate)
        net.train()
        for epoch in range(epochs):
            correct, total, epoch_loss = 0, 0, 0.0
            for images, labels in train_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = net(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                # Metrics
                epoch_loss += loss
                total += labels.size(0)
                correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            epoch_loss /= len(train_loader.dataset)
            epoch_acc = correct / total
            if verbose:
                print(f"Epoch {epoch + 1}: train loss {epoch_loss}, accuracy {epoch_acc}")

    def test(self, net: Module, val_loader: DataLoader) -> Tuple[float, float]:
        """
        Evaluate the network on the entire evaluation set.
        :param net: Neural Network.
        :param val_loader: Data loader containing the evaluation data.
        :return: A tuple containing the calculated loss and accuracy as float values.
        """
        criterion = torch.nn.CrossEntropyLoss()
        correct, total, loss = 0, 0, 0.0
        net.eval()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = net(images)
                loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        loss /= len(val_loader.dataset)
        accuracy = correct / total
        return loss, accuracy

    @staticmethod
    def set_parameters(net: Module, parameters: List[np.ndarray]) -> None:
        """
        Sets the parameters of a NN.
        :param net: Neural network.
        :param parameters: List of numpy arrays containing the parameters.
        """
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
