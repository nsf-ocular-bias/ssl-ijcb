from ffcv.writer import DatasetWriter
import numpy as np
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
    RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder

import os
import torch
from PIL import Image
import pandas as pd
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


class FairFaceDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split, labels, transform=None):
        self.data_dir = data_dir
        self.split = split
        self.transform = transform
        self.data = pd.read_csv(os.path.join(data_dir, f'fairface_label_{split}.csv'))
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.join(self.data_dir, self.data.iloc[idx].file)
        image = np.asarray(Image.open(img_name))
        label = self.labels[idx]
        # if self.transform:
        #     image = self.transform(image)
        return image, label


if __name__ == "__main__":
    
    celeb = pd.read_csv("../datasets/celeba/list_attr_celeba.csv")
    attrs = celeb.columns[1:].tolist()
    print(attrs)

    fair = pd.read_csv("../datasets/fairface/fairface_label_train.csv")

    import pickle
    labels = []

    for attr in attrs:
        with open(f"../datasets/fairface/{attr}_similarities.pkl", "rb") as f:
            data = pickle.load(f)
            data = data.argmax(1)
            # Flip 0 and 1
            data = 1 - data
            labels.append(data)

    labels = np.stack(labels, axis=1)
    print(labels.shape)

    from torch.utils.data import ConcatDataset

    ds1 = CelebaDataset("../datasets/celeba/", "train")
    ds2 = FairFaceDataset("../datasets/fairface/", "train", labels)

    ds = ConcatDataset([ds1, ds2])

    from ffcv.fields import NDArrayField, FloatField, RGBImageField, IntField
    write_path = "datasets/combined.beton"
    writer = DatasetWriter(write_path, {
        'image': RGBImageField(),
        'label': NDArrayField(shape=(40,), dtype=np.dtype('int64')),
        # 'label': IntField()

    }, num_workers=16)
    writer.from_indexed_dataset(ds)

