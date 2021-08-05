from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class FaceScrub(Dataset):
    def __init__(self, path, transform=None):
        self.root = os.path.expanduser(path)
        self.transform = transform

        input = np.load(path)

        data = input['images']
        labels = input['labels']

        #print(data.shape)
        np.random.seed(666)
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        self.data = data[perm]/255.0
        self.labels = labels[perm]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #img, target = self.data[index], self.labels[index]
        img, target = self.data[index], self.data[index]
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)

        return img, target
