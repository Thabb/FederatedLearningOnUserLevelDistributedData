import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10

"""
Loads all the data from the dataset and returns it in the form of data loaders.
"""


def load_datasets(num_clients: int, batch_size: int):
    # Download and transform CIFAR-10 (train and test)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    # TODO: Make the dataset changeable from the main file
    train_set = CIFAR10("./dataset", train=True, download=True, transform=transform)
    test_set = CIFAR10("./dataset", train=False, download=True, transform=transform)

    # Split training set into partitions to simulate the individual dataset
    partition_size = len(train_set) // num_clients
    lengths = [partition_size] * num_clients
    datasets = random_split(train_set, lengths, torch.Generator().manual_seed(42))

    for dataset in datasets:
        print(dataset)

    # Split each partition into train/val and create DataLoader
    train_loaders = []
    val_loaders = []
    for ds in datasets:
        len_val = len(ds) // 10  # 10 % validation set
        len_train = len(ds) - len_val
        lengths = [len_train, len_val]
        ds_train, ds_val = random_split(ds, lengths, torch.Generator().manual_seed(42))
        train_loaders.append(DataLoader(ds_train, batch_size=batch_size, shuffle=True))
        val_loaders.append(DataLoader(ds_val, batch_size=batch_size))
    test_loader = DataLoader(test_set, batch_size=batch_size)
    return train_loaders, val_loaders, test_loader
