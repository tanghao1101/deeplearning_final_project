import os
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms
import torch


class Imgdataset(data.Dataset):
    def __init__(self):
        super(Imgdataset, self).__init__()
        self.data = np.array(pd.read_csv('train.csv')["image_id"])
        self.image_root = 'trainimages'
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_name = self.data[index]
        img = Image.open(os.path.join(self.image_root, image_name)).convert("RGB")
        img = self.transform(img)
        return img


dataset = Imgdataset()
data_loader = data.DataLoader(dataset=dataset, batch_size=1, shuffle=False)
mean = torch.zeros(3)
std = torch.zeros(3)
for batch_idx, datas in enumerate(data_loader):
    for c in range(3):
        mean[c] += datas[:, c, :, :].mean()
        std[c] += datas[:, c, :, :].std()

mean.div_(len(dataset))
std.div_(len(dataset))
print(list(mean.numpy()), list(std.numpy()))

"""
[0.78559303, 0.7354616, 0.6978359] [0.33723018, 0.34717128, 0.36969572]
"""
