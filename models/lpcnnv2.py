import os
import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights
from torchvision import transforms
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from torchvision.transforms import ToTensor
from pytorch_lightning import loggers

"""

    LPCNN V2 - detects license plate bounding boxes in images (single-task learning problem)
    --------
    Resnet-34 -> Dense Layer with 4 outputs (licence plate coordinates)
              

"""



class LPCNNv2(pl.LightningModule):
    def __init__(self):
        super().__init__()

        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        layers = list(resnet.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.rel1 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d((1, 1))


        self.bounding_box = nn.Sequential(
            nn.BatchNorm1d(512),
            nn.Linear(512, 4)
        )

        self.toTensor = ToTensor()

    @torch.no_grad()
    def init_weights(self, m):
        print(m)
        if type(m) == nn.Conv2d:
            # print(m.weight)
            torch.nn.init.xavier_uniform_(m.weight)

    def forward(self, x):
        """
        :param x: Image (3, 500, 888)
        :return: float; (0, 1) about whether a plate was detected
        """
        print(f"input: {x.shape}")
        out = self.features1(x)
        print(f" after features 1 {out.shape}")
        out = self.features2(out)
        out = self.rel1(out)
        print(f" after features 2 {out.shape}")
        out = self.pool(out)
        print(f" after pool {out.shape}")
        out = out.view(x.shape[0], -1)
        out = self.bounding_box(out)
        print(f" after bb {out.shape}")
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
        loss = F.l1_loss(predicted_bb, target_bb[1]).sum(1)
        loss = loss.sum(1)

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        # print(loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, target_bb = batch
        predicted_bb = self(x)
        print(predicted_bb.shape)
        predicted_bb = predicted_bb.mean(axis=0)
        loss = F.l1_loss(predicted_bb, torch.flatten(target_bb[1])).sum(1)
        loss = loss.sum(1)
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


