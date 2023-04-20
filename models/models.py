import os
import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.transforms import ToTensor

"""

    LPCNN V1 - detects license plate bounding boxes in images (single-task learning problem)
    --------
    Convolutional Layer -> ReLU -> Conv. Layer -> ReLU -> MaxPool -> Dense/Fully Connected with ReLU -> Softmax (licence plate or not)

"""


class LPCNNv1(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=32,
            kernel_size=5,
        )
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        self.rel1 = nn.ReLU()

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=5,
        )
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        self.rel2 = nn.ReLU()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()
        self.dense = nn.Linear(in_features=108240, out_features=128)
        self.rel3 = nn.ReLU()

        self.bounding_box = nn.Sequential(nn.BatchNorm1d(128), nn.Linear(128, 4))

        self.toTensor = ToTensor()

    def forward(self, x):
        """

        :param x: Image (1920, 1080, 3)
        :return: float; (0, 1) about whether a plate was detected
        """
        # print(x.shape)
        out = self.rel1(self.conv1(x))
        # print(out.shape)
        out = self.rel2(self.conv2(out))
        # print(out.shape)
        out = self.pool(out)
        # print(out.shape)

        out = self.flatten(out)
        # print(out.shape)
        out = self.rel3(self.dense(out))

        # print(out.shape)
        out = self.bounding_box(out)
        # print(out.shape)
        return out

    def training_step(self, batch, batch_idx):
        # print(batch[0].shape)
        # print(batch[1])
        x, target_bb = batch
        # print(type(x))
        predicted_bb = self(x).mean(axis=0)
        print(predicted_bb)
        # print(predicted_bb.shape)
        print(target_bb[1])
        # L1 loss
        loss = F.l1_loss(predicted_bb, target_bb[1])

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        print(loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=.02)
