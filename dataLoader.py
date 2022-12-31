import numpy as np
from torch.utils import data
import os
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split
import torch
import pandas as pd

torch.manual_seed(3407)


class BitmojiDataset(data.Dataset):
    def __init__(self, mode='train'):
        super(BitmojiDataset, self).__init__()
        data = pd.read_csv('datasets/train.csv')
        trainData, testData = train_test_split(data, test_size=0.2, random_state=3407)
        # trainData, testData = data, data
        self.image_root = 'datasets/trainimages'
        self.dataset = trainData if mode == 'train' else testData
        self.transform = set_transform()
        self.images_list = np.array(self.dataset["image_id"])
        self.labels_list = np.array(self.dataset["is_male"])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image_name = self.images_list[index]
        label = self.labels_list[index]
        label = 0 if label == -1 else label
        img = Image.open(os.path.join(self.image_root, image_name)).convert("RGB")
        img = self.transform(img)
        return img, label


class TestDataset(data.Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()
        self.images_list = ["%d.jpg" % (i) for i in range(3000, 4084)]
        self.image_root = 'datasets/testimages'
        self.transform = test_transform()

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        image_name = self.images_list[index]
        img = Image.open(os.path.join(self.image_root, image_name)).convert("RGB")
        img = self.transform(img)
        return image_name, img


def set_transform():
    transform = [transforms.RandomCrop(size=320), transforms.Resize(size=224), transforms.ToTensor(),
                 transforms.RandomHorizontalFlip(p=0.5), transforms.Normalize(mean=[0.78559303, 0.7354616, 0.6978359],
                                                                              std=[0.33723018, 0.34717128, 0.36969572])]
    transform = transforms.Compose(transform)
    return transform


def test_transform():
    transform = [transforms.Resize(size=224), transforms.ToTensor(),
                 transforms.Normalize(mean=[0.78559303, 0.7354616, 0.6978359],
                                      std=[0.33723018, 0.34717128, 0.36969572])]
    transform = transforms.Compose(transform)
    return transform


def dataloader(batch_size, mode="train", true_test=False):
    if not true_test:
        dataset = BitmojiDataset(mode)
    else:
        dataset = TestDataset()
    data_loader = data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=(mode == "train"), num_workers=0)
    return data_loader
