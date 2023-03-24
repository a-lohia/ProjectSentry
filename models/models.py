import os
import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl


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
        self.rel2 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.dense = nn.Softmax()


    def __call__(self, x):
        return
