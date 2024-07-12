"""
Train a model on the FairFace dataset using supervised learning with lightning.
"""

import torch
import pytorch_lightning as pl
from torch.optim import optimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from dataclasses import dataclass
from ffcv.writer import DatasetWriter
import numpy as np
import ffcv
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from loss import SupConLoss
from ffcv.fields.ndarray import NDArrayDecoder

from sklearn.linear_model import LogisticRegression
 
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256
res = 224

@dataclass(frozen=True)
class Hyperparameters:
    data_dir: str ="../datasets/fairface/"
    batch_size: int = 256
    num_classes: int = 2
    learning_rate: float = 3e-4
    max_epochs: int = 300


class CombinedSupCon(pl.LightningModule):
    def __init__(self, hparams: Hyperparameters):
        super().__init__()
        self.save_hyperparameters()
        self.params = hparams
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)
        # self.model.fc = torch.nn.Linear(512, self.params.num_classes)
        self.model.fc = torch.nn.Sequential(
            torch.nn.Linear(512, 2048),
            torch.nn.BatchNorm1d(2048),
            torch.nn.ReLU(True),
            torch.nn.Linear(2048, 512, bias=False),
        )
        self.loss = SupConLoss(temperature=0.1)

        self.linear_probe = LogisticRegression(max_iter=1000)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x1, y, x2 = batch
        x = torch.cat([x1, x2], dim=0)
        y_hat = self(x)
        y_hat = torch.nn.functional.normalize(y_hat, dim=1)
        bsz = y.size(0)
        f1, f2 = torch.split(y_hat, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

        # loss = [None] * 40
        # for i in range(40):
        #     loss[i] = self.loss(features, y[:, i])
        loss = [self.loss(features, y[:, i]) for i in range(40)]
        loss = torch.stack(loss).mean()
        self.log('train_loss', loss, sync_dist=True,prog_bar=True)

        if batch_idx == 0:
            self.linear_probe = LogisticRegression(max_iter=1000)
            self.linear_probe.fit(f1.cpu().detach().numpy(), y.cpu().detach().numpy()[:, 20])
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)


        # Take Random Projection to 2 dimension
        y_hat = self.linear_probe.predict(y_hat.cpu().detach().numpy())

        self.log('val_loss', loss, sync_dist=True)
        # Log the accuracy
        # preds = torch.argmax(torch.Tensor(y_hat), dim=1)
        # acc = (y_hat == y).float().mean()
        acc = (torch.Tensor(y_hat).to(y.device) == y).float().mean()
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.learning_rate)
        # optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-6)
        learning_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.params.max_epochs, verbose=True)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': learning_scheduler,
            }
        }

    def train_dataloader(self):
        decoder  = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline_big = [
            decoder,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ToTensor(),
            ToDevice(torch.device(self.trainer.local_rank), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2))
        ]

        # Second branch of augmentations
            
        decoder2  = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline_big2= [
            decoder2,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomColorJitter(0.8, 0.4, 0.4, 0.2, 0.1),
            ffcv.transforms.RandomGrayscale(0.2),
            ffcv.transforms.RandomSolarization(0.2, 128),
            ToTensor(),
            ToDevice(torch.device(self.trainer.local_rank), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        # SSL Augmentation pipeline
        label_pipeline = [
            # IntDecoder(),
            NDArrayDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.trainer.local_rank), non_blocking=True)
        ]

        decoder = RandomResizedCropRGBImageDecoder((res, res))
        distributed = True
        batch_size = 256
        num_workers = 10
        in_memory = True
        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        train_loader = Loader("datasets/combined.beton",
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline_big,
                            'label': label_pipeline,
                            'image_0': image_pipeline_big2,
                        },
                        custom_field_mapper= {"image_0": 'image'},
                        distributed=distributed)
        return train_loader
    
    def val_dataloader(self):
        
        cropper = CenterCropRGBImageDecoder((224, 224), ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(torch.device(self.trainer.local_rank), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.trainer.local_rank),  non_blocking=True)
        ]

        val_loader = Loader("datasets/fairface_val.beton",
                        batch_size=256,
                        num_workers=2,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=True)
        return val_loader




def main():
    hparams = Hyperparameters()
     # Callbacks
    model_ckpt = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_last=True)

    trainer = pl.Trainer(num_sanity_val_steps=0, max_epochs=hparams.max_epochs, devices=[0, 1], precision="16-mixed", benchmark=True, callbacks=[model_ckpt], profiler=None, strategy="ddp", accelerator="gpu")

    model = CombinedSupCon(hparams)
    

    trainer.fit(model)


if __name__ == "__main__":
    main()
