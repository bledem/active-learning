"""MNIST DataModule"""
import argparse
import numpy as np

from torch.utils.data import random_split, Subset
from torchvision.datasets import MNIST as TorchMNIST
from torchvision import transforms

from text_recognizer.data.base_data_module import BaseDataModule, load_and_print_info

DOWNLOADED_DATA_DIRNAME = BaseDataModule.data_dirname() / "downloaded"

# NOTE: temp fix until https://github.com/pytorch/vision/issues/1938 is resolved
from six.moves import urllib  # pylint: disable=wrong-import-position, wrong-import-order

opener = urllib.request.build_opener()
opener.addheaders = [("User-agent", "Mozilla/5.0")]
urllib.request.install_opener(opener)


class MNIST(BaseDataModule):
    """
    MNIST DataModule.
    Learn more at https://pytorch-lightning.readthedocs.io/en/stable/extensions/datamodules.html
    """

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.data_dir = DOWNLOADED_DATA_DIRNAME
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.dims = (1, 28, 28)  # dims are returned when calling `.size()` on this object.
        self.output_dims = (1,)
        self.mapping = list(range(10))

    def prepare_data(self, *args, **kwargs) -> None:
        """Download train and test MNIST data from PyTorch canonical source."""
        TorchMNIST(self.data_dir, train=True, download=True)
        TorchMNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None, split_train_val=0.15, split_test=1) -> None:
        """Split into train, val, test, and set dims."""
        mnist_full = TorchMNIST(self.data_dir, train=True, transform=self.transform)
        self.data_train, self.data_val = random_split(mnist_full, [55000, 5000])  # type: ignore
        self.data_train = Subset(self.data_train, np.arange(int(len(self.data_train)*split_train_val)))
        self.data_val = Subset(self.data_val, np.arange(int(len(self.data_val)*split_train_val)))
        self.data_test = TorchMNIST(self.data_dir, train=False, transform=self.transform)
        self.data_test = Subset(self.data_test, np.arange(int(len(self.data_test)*split_test)))


if __name__ == "__main__":
    load_and_print_info(MNIST)
