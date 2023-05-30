import os
import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import torchvision.ops.diou_loss as D
import torchvision.ops.ciou_loss as C
import torchvision.ops.boxes as boxes
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
        inceptionv3 = torch.hub.load('pytorch/vision:v0.10.0', 'inception_v3', weights="Inception_V3_Weights.DEFAULT")
        # resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', weights='ResNet34_Weights.DEFAULT')
        layers = list(inceptionv3.children())[:8]
        self.features1 = nn.Sequential(*layers[:6])
        self.features2 = nn.Sequential(*layers[6:])
        self.rel1 = nn.ReLU()

        self.pool = nn.AdaptiveAvgPool2d((10, 10))  # (10, 10)

        self.bounding_box = nn.Sequential(
            nn.BatchNorm1d(25600),
            nn.Linear(25600, 4)
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
        # print(f"input: {x.shape}")
        out = self.features1(x)
        # print(f" after features 1 {out.shape}")
        out = self.features2(out)
        out = self.rel1(out)
        # print(f" after features 2 {out.shape}")
        out = self.pool(out)
        # print(f" after pool {out.shape}")
        out = out.view(x.shape[0], -1)
        # print(f" after view change {out.shape}")
        out = self.bounding_box(out)
        # print(f" after bb {out.shape}")
        return out

    def training_step(self, batch, batch_idx):
        # print(batch[0].shape)
        # print(f"batch[1]: {batch[1]}")
        x, target_bb = batch
        # print(type(x))
        # print(self(x).shape)
        predicted_bb = self(x)
        # print(predicted_bb)
        # print(predicted_bb.shape)
        # print(target_bb[1])
        # L1 loss
        loss1 = F.smooth_l1_loss(predicted_bb, target_bb[1], reduction="sum")
        # F.mse_loss(predicted_bb, target_bb[1], reduction="mean")
        # print(loss1)
        loss2 = D.distance_box_iou_loss(predicted_bb, target_bb[1], reduction="sum")
        # loss2 = C.complete_box_iou_loss(predicted_bb, target_bb[1], reduction="mean")
        loss = loss1/100 + loss2
        print(f"total loss: {loss}")

        accuracy = boxes.box_iou(predicted_bb, target_bb[1]).mean()

        # logs metrics for each training_step,
        # and the average across the epoch, to the progress bar and logger
        # print(loss)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0])
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=x.shape[0])
        return loss

    def test_step(self, batch, batch_idx):
        x, target_bb = batch
        predicted_bb = self(x)

        loss = F.l1_loss(predicted_bb, target_bb[1])
        accuracy = boxes.complete_box_iou(predicted_bb, target_bb[1]).mean()
        self.log("test_accuracy", accuracy)
        self.log("test_loss", loss)

    def validation_step(self, batch, batch_idx):
        x, target_bb = batch
        predicted_bb = self(x)

        loss = F.l1_loss(predicted_bb, target_bb[1])
        accuracy = boxes.box_iou(predicted_bb, target_bb[1]).mean()
        self.log("val_accuracy", accuracy)
        self.log("val_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=.02)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        tensorboard = self.logger.experiment.add_scalar("Loss/Train", outputs["loss"], batch_idx)
        batch_dictionary = {
            'loss': outputs["loss"]
        }
        return batch_dictionary
