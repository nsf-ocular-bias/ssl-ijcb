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
from ffcv.fields.ndarray import NDArrayDecoder
 
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256
res = 224

@dataclass(frozen=True)
class Hyperparameters:
    data_dir: str ="../datasets/celeba/"
    batch_size: int = 256
    num_classes: int = 40
    learning_rate: float = 3e-4
    max_epochs: int = 100



class CelebaDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.df = pd.read_csv(os.path.join(data_dir, "list_attr_celeba.csv"))
        self.eval_partition = pd.read_csv(os.path.join(data_dir, "list_eval_partition.csv"))

        if split == "train":
            self.data = self.df[self.eval_partition["partition"] == 0]
        elif split == "test":
            self.data = self.df[self.eval_partition["partition"] == 2]
        elif split == "val":
            self.data = self.df[self.eval_partition["partition"] == 1]
        else:
            raise ValueError("Invalid split")



    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, "img_align_celeba/img_align_celeba", self.data.iloc[idx].image_id)
        image = np.asarray(Image.open(img_name))
        labels = [self.data.iloc[idx][x] for x in self.data.columns[1:]]
        labels = [1 if x == 1 else 0 for x in labels]
        # if self.transform:
        #     image = self.transform(image)
        return image, np.array(labels, dtype=np.int64)
        # # Get just the gender label
        # return image, labels[21]

class CelebaModel(pl.LightningModule):
    def __init__(self, hparams: Hyperparameters):
        super().__init__()
        self.save_hyperparameters()
        self.params = hparams
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=False)
        # self.model.fc = torch.nn.Linear(512, self.params.num_classes)
        self.model.fc = torch.nn.Identity()

        self.model.fcs = torch.nn.ModuleList([torch.nn.Linear(512, 2) for _ in range(40)])

    def forward(self, x):
        x = self.model(x)
        return torch.stack([fc(x) for fc in self.model.fcs], dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        losses = [torch.nn.functional.cross_entropy(y_hat[i, :], y[i]) for i in range(40)]
        loss = torch.stack(losses).mean()
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        losses = [torch.nn.functional.cross_entropy(y_hat[i, :], y[i]) for i in range(40)]
        loss = torch.stack(losses).mean()
        self.log('val_loss', loss, sync_dist=True)
        # Log the accuracy
        preds = [torch.argmax(y_hat[i, :], dim=1) for i in range(40)]
        accs = [(preds[i] == y[i]).float().mean() for i in range(40)]
        for i in range(40):
            self.log(f'val_acc_{i}', accs[i], prog_bar=False, sync_dist=True)
        acc = torch.stack(accs).mean()
        self.log('val_acc_mean', acc, prog_bar=True, sync_dist=True)
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
            NDArrayDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.trainer.local_rank), non_blocking=True)
        ]

        distributed = True
        batch_size = 256
        num_workers = 8
        in_memory = True
        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        # train_loader = Loader("datasets/celeba_train.beton",
        train_loader = Loader("datasets/combined.beton",
                        batch_size=batch_size,
                        indices = np.arange(249514)[162770:],
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed)
        print(train_loader.indices.shape)
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
            NDArrayDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(self.trainer.local_rank),  non_blocking=True)
        ]

        val_loader = Loader("datasets/celeba_val.beton",
                        batch_size=256,
                        num_workers=2,
                        order=OrderOption.SEQUENTIAL,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'labels': label_pipeline
                        },
                        distributed=True)
        return val_loader




def main():
    hparams = Hyperparameters()
     # Callbacks
    model_ckpt = pl.callbacks.ModelCheckpoint(monitor='val_loss', save_last=True)

    trainer = pl.Trainer(max_epochs=hparams.max_epochs, devices=[0, 1], precision="16-mixed", benchmark=True, callbacks=[model_ckpt], profiler=None, strategy="ddp", accelerator="gpu")

    model = CelebaModel(hparams)
    

    trainer.fit(model)


if __name__ == "__main__":
    main()


    # ds = CelebaDataset("../datasets/celeba/", "test")
    # from ffcv.fields import NDArrayField, FloatField, RGBImageField, IntField
    # write_path = "celeba_test_gender.beton"
    # writer = DatasetWriter(write_path, {
    #     'image': RGBImageField(),
    #     'labels': NDArrayField(shape=(40,), dtype=np.dtype('int64')),
    #     # 'label': IntField()
    #
    # }, num_workers=16)
    # writer.from_indexed_dataset(ds)
