#dataset to train the autoencoder

from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class FaceScrub(Dataset):
    def __init__(self, path, transform=None):
        self.root = os.path.expanduser(path)
        self.transform = transform

        dataset = np.load(path)

        data = dataset['images']
        labels = dataset['labels']

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

class Celeb(Dataset):
    def __init__(self, path, transform=None):
        self.root = os.path.expanduser(path)
        self.transform = transform

        data = np.load(path)

        #print(data.shape)
        np.random.seed(666)
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        self.data = data[perm]/255.0

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

 class Fawkes(Dataset):
    def __init__(self, path, transform=None):
        #self.root = os.path.expanduser(path)
        self.path = path
        self.transform = transform

        file = os.path.join(self.path, 'fawkes.npz')
        dataset = np.load(file)
        #we are gonna to recover the image so the images are labels
        labels = dataset['images']
        data = dataset['fawkes']
        '''
        #print(data.shape)
        np.random.seed(666)
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        '''

        self.data = data/255.0
        self.label = labels/225.0

    #save original images with reconstructed images
    def save_recon(self, recon):
        file = os.path.join(self.path, 'fawkes_recon.npy') 
        np.save(file, recon)
        print('saved as {}'.format(file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        #img, target = self.data[index], self.labels[index]
        img, target = self.data[index], self.label[index]
        #img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
            target = self.transform(target)

        return img, target


