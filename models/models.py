import os
import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl

"""

    LPCNN V1 - detects license plate bounding boxes in images (single-task learning problem)
    --------
    Convolutional Layer -> ReLU -> Conv. Layer -> ReLU -> MaxPool -> Dense/Fully Connected with ReLU -> Softmax (licence plate or not)

"""
class LPCNNv1(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=5,
        )
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.rel1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
        )
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.rel2 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=, out_features=)
        self.rel3 = nn.ReLU()

        self.softmax = nn.Softmax()


    def __call__(self, x):
        """

        :param x: Image (1920, 1080, 3)
        :return: float; (0, 1) about whether a plate was detected
        """
        out = self.rel1(self.conv1(x))
        out = self.rel2(self.conv2(out))
        out = self.pool(out)
        out = self.rel3(self.dense(self.flatten(out)))
        out = self.softmax(out)
        return out
