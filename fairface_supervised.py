"""
Train a model on the FairFace dataset using supervised learning with lightning.
"""

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from dataclasses import dataclass
from ffcv.writer import DatasetWriter
import numpy as np
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
 
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
    max_epochs: int = 100


class FairFaceDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.data = pd.read_csv(os.path.join(data_dir, f'fairface_label_{split}.csv'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.data.iloc[idx].file)
        image = np.asarray(Image.open(img_name))
        label = self.data.iloc[idx].gender
        label = 0 if label == "Female" else 1
        # if self.transform:
        #     image = self.transform(image)
        return image, label

class FairFaceModel(pl.LightningModule):
    def __init__(self, hparams: Hyperparameters):
        super().__init__()
        self.save_hyperparameters()
        self.params = hparams
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
        self.model.fc = torch.nn.Linear(512, self.params.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        self.log('val_loss', loss, sync_dist=True)
        # Log the accuracy
        preds = torch.argmax(y_hat, dim=1)
        acc = (preds == y).float().mean()
        self.log('val_acc', acc, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.learning_rate)
        return optimizer

    def train_dataloader(self):
        decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline= [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(torch.device(self.trainer.local_rank), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.trainer.local_rank), non_blocking=True)
        ]

        distributed = True
        batch_size = 256
        num_workers = 8
        in_memory = True
        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        train_loader = Loader("fairface_train.beton",
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
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

        val_loader = Loader("fairface_val.beton",
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
    model_ckpt = pl.callbacks.ModelCheckpoint(monitor='val_loss')

    trainer = pl.Trainer(max_epochs=hparams.max_epochs, devices=[0, 1], precision="16-mixed", benchmark=True, callbacks=[model_ckpt], profiler=None, strategy="ddp", accelerator="gpu")

    model = FairFaceModel(hparams)
    

    trainer.fit(model)


if __name__ == "__main__":
    main()


    # ds = FairFaceDataset("../datasets/fairface/", "val")
    # from ffcv.fields import NDArrayField, FloatField, RGBImageField, IntField
    # write_path = "fairface_val.beton"
    # writer = DatasetWriter(write_path, {
    #     'image': RGBImageField(),
    #     'label': IntField(),
    #
    # }, num_workers=16)
    # writer.from_indexed_dataset(ds)
