from torch.utils.data import Dataset
from torchvision.datasets import mnist

from .base import BasicDataset


class MNIST(BasicDataset):
    name = "mnist"
    num_classes = 10
    shape = (1, 1, 28, 28)

    mean = (0.1307,)
    std_dev = (0.3081,)
    num_train_samples = 60000
    num_val_samples = 10000

    def get_dataset(self, download: bool = True) -> Dataset:
        return mnist.MNIST(
            root=self.root_directory,
            train=self.is_train,
            transform=self.get_transform(),
            download=download,
        )
