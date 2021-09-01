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

#fawkes dataset for reconstruction use
#path is the folder path that contains dataset
class Fawkes(Dataset):
    def __init__(self, path, transform=None, shuffle = False):
        #self.root = os.path.expanduser(path)
        self.path = path
        self.transform = transform

        #file = os.path.join(self.path, 'fawkes.npz')
        dataset = np.load(self.path)

        #we are gonna to recover the image so the images are labels
        labels = dataset['images']
        data = dataset['fawkes']
        
        #print(data.shape)
        np.random.seed(666)
        perm = np.arange(len(data))
        np.random.shuffle(perm)


        if shuffle:

            self.data = data[perm]/255.0
            self.label = labels[perm]/225.0
        else:
            self.data = data/255.0
            self.label = labels/225.0

    #save original images with reconstructed images
    def save_recon(self, recon, msg = None):
        file = self.path[:-4]+ msg +'_recon.npy'
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

#fawkes dataset for training and validation
#full path of dataset
class Fawkes_train(Dataset):
    def __init__(self, path, transform=None, train  = True, train_size = 0.8):
        #self.root = os.path.expanduser(path)
        self.path = path
        self.transform = transform

        file = os.path.join(self.path)
        dataset = np.load(file)

        #we are gonna to recover the image so the images are labels
        labels = dataset['images']
        data = dataset['fawkes']

        idx = int(data.shape[0] * train_size)

        if train == True:
            self.data = data[:idx]/255.0
            self.label = labels[:idx]/225.0
        else:
            #for test
            self.data = data[idx:]/255.0
            self.label = labels[idx:]/225.0



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
