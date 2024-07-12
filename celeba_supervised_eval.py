import torchmetrics
import torchvision.models.resnet as resnet
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
from dataclasses import dataclass
import ffcv
from ffcv.writer import DatasetWriter
import numpy as np
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage, RandomColorJitter, RandomGrayscale
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from ffcv.fields.ndarray import NDArrayDecoder
from torchvision.transforms import AutoAugment
 
from loss import SupConLoss


@dataclass(frozen=True)
class Hyperparameters:
    data_dir: str ="../datasets/fairface/"
    batch_size: int = 256
    num_classes: int = 2
    learning_rate: float = 3e-4
    max_epochs: int = 100


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256
res = 224
Idx = 6

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

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, "img_align_celeba/img_align_celeba", self.data.iloc[idx].image_id)
        image = Image.open(img_name)
        labels = [self.data.iloc[idx][x] for x in self.data.columns[1:]]
        labels = [1 if x == 1 else 0 for x in labels]
        if self.transform:
            image = self.transform(image)
        return image, labels[Idx], labels[20]
        # # Get just the gender label
        # return image, labels[21]

def main():
    import celeba_supervised
    hparams = celeba_supervised.Hyperparameters()
    early_stopping = pl.callbacks.EarlyStopping(monitor="val_f1", mode="max", patience=12, verbose=True)
    model_ckpt = pl.callbacks.ModelCheckpoint(monitor='val_f1', mode='max')

    trainer = pl.Trainer(max_epochs=hparams.max_epochs, devices=[0, 1], precision="16-mixed", benchmark=True, callbacks=[model_ckpt, early_stopping], profiler=None, strategy="ddp", accelerator="gpu")
    from celeba_supervised import CelebaModel
    model = CelebaModel.load_from_checkpoint("lightning_logs/version_200/checkpoints/last.ckpt", hparams=hparams)
    # model = FairFaceModel(hparams)
    

    # trainer.fit(model)
    #
    # if trainer.local_rank != 0:
    #     return
    # print("Training Done")
    #
    # print(f"Best Model: {model_ckpt.best_model_path}")
    #
    # # Load the best model
    # model = FairFaceModel.load_from_checkpoint(model_ckpt.best_model_path)
 
    # Test the model
    test_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN / 255, IMAGENET_STD/ 255)
    ])

    test_dataset = CelebaDataset("../datasets/celeba/", "train", transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False, num_workers=8)

    model.fc = torch.nn.Identity()

    model.eval()
    model = model.to("cuda")

    preds = []
    actual = []
    rgs = []
    
    feats = []
    from tqdm import tqdm
    for x, y, rg in tqdm(test_loader):
        x = x.to("cuda")
        y = y.to("cuda")
        with torch.no_grad():
            pred = model(x)[:,Idx,:]
            feats.append(pred)
            pred = torch.softmax(pred, dim=1)
            pred = torch.argmax(pred, dim=1)
            preds.append(pred)
            actual.append(y)
            rgs.extend(rg)
    preds = torch.cat(preds).cpu().numpy()
    actual = torch.cat(actual).cpu().numpy()
    rgs = np.array(rgs)

    # feats = torch.cat(feats)

    # import pickle
    # with open("feats_129_train.pkl", 'wb') as f:
    #     # pickle.dump([preds, actual, rgs], f)
    #     pickle.dump(feats, f)


    
    # Calculate Accuracy
    acc = torchmetrics.functional.accuracy(torch.tensor(preds), torch.tensor(actual), task='binary').item()
    print(f"Accuracy: {acc*100:.2f}")

    # Calculate Accuracy of unique rgs
    unique_rgs = np.unique(rgs)
    accs = []
    for rg in unique_rgs:
        idx = np.where(rgs == rg)
        acc = torchmetrics.functional.accuracy(torch.tensor(preds[idx]), torch.tensor(actual[idx]), task='binary').item()
        accs.append(acc)

    std = np.std(accs)

    print(f"STD: {std*100:.2f}")

    # Calculate Min and Max Accuracy
    min_acc = np.min(accs)
    max_acc = np.max(accs)

    print(f"Min Accuracy: {min_acc*100:.2f}")
    print(f"Max Accuracy: {max_acc*100:.2f}")

    # Calculate SeR, Selection Rate (Min/Max)
    ser = min_acc / max_acc

    print(f"SeR: {ser*100:.2f}")

    # Calculate Demographic Parity from aif360
    from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio, equalized_odds_difference

    dpd = demographic_parity_difference(y_true=actual, y_pred=preds, sensitive_features=rgs)
    dpr = demographic_parity_ratio(y_true=actual, y_pred=preds, sensitive_features=rgs)
    eod  = equalized_odds_difference(y_true=actual, y_pred=preds, sensitive_features=rgs)

    print(f"DPD: {dpd*100:.2f}")

    print(f"DPR: {dpr*100:.2f}")
    print(f"EOD: {eod*100:.2f}")






if __name__ == "__main__":
    main()
