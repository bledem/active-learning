from typing import Any, Dict
import argparse

from torchvision.models import resnet34, resnet18, resnet101
from torch import nn
import torch

IMAGE_SIZE = 28

class Resnet18(nn.Module):
    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dims = data_config["input_dims"]
        num_classes = len(data_config["mapping"])
        pretrained = False #bool(self.args['pretrained']==1)
        self.model = resnet18(pretrained=pretrained, num_classes=num_classes) # updating the nb of classes
        # updating the number of input channel
        self.model.conv1 = nn.Conv2d(input_dims[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # updating last fc layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        _B, _C, H, W = x.shape
        assert H == W == IMAGE_SIZE
        x = self.model(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrained", type=int, default=1)
        return parser

class Resnet34(nn.Module):
    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dims = data_config["input_dims"]
        num_classes = len(data_config["mapping"])
        pretrained = False #bool(self.args['pretrained']==1)
        self.model = resnet34(pretrained=pretrained, num_classes=num_classes) # updating the nb of classes
        # updating the number of input channel
        self.model.conv1 = nn.Conv2d(input_dims[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # updating last fc layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        _B, _C, H, W = x.shape
        assert H == W == IMAGE_SIZE
        x = self.model(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrained", type=int, default=1)
        return parser

class Resnet101(nn.Module):
    def __init__(self, data_config: Dict[str, Any], args: argparse.Namespace = None) -> None:
        super().__init__()
        self.args = vars(args) if args is not None else {}

        input_dims = data_config["input_dims"]
        num_classes = len(data_config["mapping"])
        pretrained = False #bool(self.args['pretrained']==1)
        self.model = resnet101(pretrained=pretrained, num_classes=num_classes) # updating the nb of classes
        # updating the number of input channel
        self.model.conv1 = nn.Conv2d(input_dims[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # updating last fc layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
        x
            (B, C, H, W) tensor, where H and W must equal IMAGE_SIZE

        Returns
        -------
        torch.Tensor
            (B, C) tensor
        """
        _B, _C, H, W = x.shape
        assert H == W == IMAGE_SIZE
        x = self.model(x)
        return x

    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--pretrained", type=int, default=1)
        return parser