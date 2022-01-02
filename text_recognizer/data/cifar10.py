import os
import torch
import h5py

import numpy as np
from text_recognizer.data.base_data_module import BaseDataModule
from torchvision import transforms as transform_lib
from torchvision.datasets import CIFAR10
from torch.utils.data import random_split
import pl_bolts
from torch.utils.data import Subset
from text_recognizer.data.util import BaseDataset, split_dataset

PROCESSED_DATA_DIRNAME = BaseDataModule.data_dirname() / "processed" / "cifar10"
TRAIN_FRAC = 0.9

class CIFAR10DataModule(BaseDataModule):

    name = 'cifar10'
    extra_args = {}

    def __init__(
            self, data_path, args,
            val_split: int = 5000,
            num_workers: int = 16,
            batch_size: int = 40,
            split_ratio=1,
            seed: int = 42):
        """
        .. figure:: https://3qeqpr26caki16dnhd19sv6by6v-wpengine.netdna-ssl.com/wp-content/uploads/2019/01/
            Plot-of-a-Subset-of-Images-from-the-CIFAR-10-Dataset.png
            :width: 400
            :alt: CIFAR-10
        Specs:
            - 10 classes (1 per class)
            - Each image is (3 x 32 x 32)
        Standard CIFAR10, train, val, test splits and transforms
        Transforms::
            mnist_transforms = transform_lib.Compose([
                transform_lib.ToTensor(),
                transforms.Normalize(
                    mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                    std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
                )
            ])
        Example::
            from pl_bolts.datamodules import CIFAR10DataModule
            dm = CIFAR10DataModule(PATH)
            model = LitModel()
            Trainer().fit(model, dm)
        Or you can set your own transforms
        Example::
            dm.train_transforms = ...
            dm.test_transforms = ...
            dm.val_transforms  = ...
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            batch_size: number of examples per training/eval step
        """
        super().__init__(args)
        self.dims = (3, 32, 32)
        self.output_dims = (1,)
        self.DATASET = CIFAR10
        self.val_split = val_split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.seed = seed
        self.data_dir = data_path if data_path is not None else os.getcwd()
        self.num_samples = 60000 - val_split
        self.split_train_val_ratio = split_ratio
        self.data_path = data_path
        self.mapping = list(range(0,10)) #TOFILL
        self.transform = transform_lib.Compose([transform_lib.ToTensor()])

        self.prepare_data()

    @property
    def num_classes(self):
        """
        Return:
            10
        """
        return 10

    def read_h5(self):
        with h5py.File(self.data_path, "r") as f:
            self.x_trainval = f["x_train"][:]
            self.y_trainval = f["y_train"][:].squeeze().astype(int)
            if 'x_val' in f:
                self.x_val = f["x_val"][:]
                self.y_val = f["y_val"][:].squeeze().astype(int)
            else:
                self.x_val, self.y_val = None, None
            self.x_test = f["x_test"][:]
            self.y_test = f["y_test"][:].squeeze().astype(int)
        
    def prepare_data(self):
        """
        Saves CIFAR10 files to data_dir
        """
        if self.data_dir == PROCESSED_DATA_DIRNAME:
            self.DATASET(PROCESSED_DATA_DIRNAME, train=True, download=True, transform=transform_lib.ToTensor(), **self.extra_args)
            self.DATASET(PROCESSED_DATA_DIRNAME, train=False, download=True, transform=transform_lib.ToTensor(), **self.extra_args)
        else:
            self.read_h5()

    def setup(self, split_train_val=None):
        split_train_val_ratio = split_train_val if split_train_val else \
            self.split_train_val_ratio
        # original dataset
        if self.data_dir == PROCESSED_DATA_DIRNAME:
            # test data
            transforms = self.default_transforms() if self.test_transforms is None else self.test_transforms

            self.data_test = self.DATASET(PROCESSED_DATA_DIRNAME, train=False, download=False, transform=transforms, **self.extra_args)
            # train data
            transforms = self.default_transforms() if self.train_transforms is None else self.train_transforms
            dataset = self.DATASET(PROCESSED_DATA_DIRNAME, train=True, download=False, transform=transforms, **self.extra_args)
            # dataset.data (50000, 32, 32, 3) (array), ataset.targets (5000) (list)
            
            # transposed_data = np.transpose(dataset.data, (0,3,1,2))
            data_trainval = BaseDataset(dataset.data, np.array(dataset.targets), transform=self.transform)
            self.data_train, self.data_val = split_dataset(base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=42)
            self.data_train_complete = self.data_train
            self.data_train = Subset(self.data_train, np.arange(int(len(self.data_train)*split_train_val_ratio)))
        # sampled dataset
        else:
            split_train_val_ratio = 1
            data_trainval = BaseDataset(self.x_trainval, self.y_trainval, transform=self.transform)
            # if the dataset has been enriched from sampling, we keep same val data
            if  self.x_val is not None:
                self.data_train = data_trainval
                self.data_val = BaseDataset(self.x_val, self.y_val, transform=self.transform)
                # simulating the split_dataet behaviour
                self.data_val = Subset(self.data_val, np.arange(int(len(self.data_val))))
                self.data_train = Subset(self.data_train, np.arange(int(len(self.data_train))))
            else:
                self.data_train, self.data_val = split_dataset(base_dataset=data_trainval, fraction=TRAIN_FRAC, seed=42)
            # self.data_val = Subset(self.data_val, np.arange(int(len(self.data_val)*split_train_val)))
            self.data_test = BaseDataset(self.x_test, self.y_test, transform=self.transform)
    

    def default_transforms(self):
        cf10_transforms = transform_lib.Compose([
            transform_lib.ToTensor(),
            pl_bolts.transforms.dataset_normalizations.cifar10_normalization()
        ])
        return cf10_transforms