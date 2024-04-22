"""
This submodule contains data preparation code for some of the datasets used with our models,
i.e. MNIST, CIFAR 10 and 100 and ImageNet.
"""

from typing import List, Type

from .base import BasicDataset
from .mnist import MNIST

__all__ = [
    "BasicDataset",
    "dataset_from_name",
    "dataset_names",
    "MNIST",
]


def dataset_from_name(name: str) -> Type[BasicDataset]:
    """returns the dataset to which the name belongs to (name has to be the value of the datasets
    name-attribute)

    Args:
        name (str): name of the dataset

    Raises:
        ValueError: raised if no dataset under that name was found

    Returns:
        dataset: the dataset
    """
    for dataset_class in [MNIST]:
        if dataset_class.name == name:
            return dataset_class
    raise Exception(f"unknown dataset: {name}")


def dataset_names() -> List[str]:
    """getter for list of dataset names for argparse

    Returns:
        List: the dataset names
    """
    return [dataset_class.name for dataset_class in [MNIST]]
