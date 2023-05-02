import os
import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.transforms import ToTensor
from pytorch_lightning import loggers

"""

    LPCNN V1 - detects license plate bounding boxes in images (single-task learning problem)
    --------
    Convolutional Layer -> ReLU -> Conv. Layer -> ReLU -> MaxPool -> Dense/Fully Connected with ReLU -> Softmax (licence plate or not)

"""


class LPCNNv1(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=5,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=32,
                kernel_size=5,
                stride=2
            ),
            nn.ReLU()
        )
        self.conv.apply(self.init_weights)

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Sequential(
            nn.Linear(in_features=865920, out_features=4096),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(in_features=4096, out_features=128),
            nn.ReLU()
        )

        self.bounding_box = nn.Sequential(
            nn.BatchNorm1d(128),
            nn.Linear(128, 4)
        )

        self.toTensor = ToTensor()

    @torch.no_grad()
    def init_weights(self, m):
        # print(m)
        if type(m) == nn.Conv2d:
            # print(m.weight)
            torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """
        :param x: Image (3, 500, 888)
        :return: float; (0, 1) about whether a plate was detected
        """
        print(x.shape)
        out = self.conv(x)
        print(out.shape)
        out = self.pool(out)
        print(out.shape)

        out = self.flatten(out)
        print(out.shape)
        out = self.fc1(out)
        print(out.shape)
        out = self.fc2(out)

        print(out.shape)
        out = self.bounding_box(torch.flatten(out))
        print(out.shape)
        return out

    def training_step(self, batch, batch_idx):
        # print(batch[0].shape)
        # print(batch[1])
        x, target_bb = batch
        # print(type(x))
        predicted_bb = torch.flatten(self(x).mean(axis=0))
        # print(predicted_bb)
        # print(predicted_bb.shape)
        # print(target_bb[1])
        # L1 loss
        loss = F.l1_loss(predicted_bb, target_bb[1])

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        # print(loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, target_bb = batch
        predicted_bb = self(x).mean(axis=0)
        loss = F.l1_loss(predicted_bb, torch.flatten(target_bb[1]))
        self.log("test_loss", loss)

    def validation_step(self, batch, batch_idx):
        x, target_bb = batch
        predicted_bb = self(x).mean(axis=0)
        loss = F.l1_loss(predicted_bb, target_bb[1])
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=.02)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        tensorboard = self.logger.experiment.add_scalar("Loss/Train", outputs["loss"], batch_idx)
        batch_dictionary = {
            'loss': outputs["loss"]
        }
        return batch_dictionary


