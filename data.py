#dataset to train the autoencoder

from __future__ import print_function, division
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from skimage.restoration import (denoise_wavelet, estimate_sigma, 
                                calibrate_denoiser, denoise_nl_means,
                                denoise_tv_chambolle, denoise_bilateral)

from utils import get_feature
from resnet import get_feature_resnet

def tv(images, weight = 0.3):
    data = []

    for noisy in images:

        denoise = denoise_tv_chambolle(noisy, weight=weight, multichannel = True)
        data.append(denoise)

    print('=================tv chambolle denoising done!=============')
    
    return np.array(data)

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
    def __init__(self, path, transform=None, train = True, train_size = 0.8):
        self.root = os.path.expanduser(path)
        self.transform = transform

        data = np.load(path)

        #print(data.shape)
        '''
        np.random.seed(666)
        perm = np.arange(len(data))
        np.random.shuffle(perm)
        '''
        labels = data
        #print(data.shape)
        idx = int(data.shape[0] * train_size)
        #print(idx)
        if train:
            self.data = tv(data[:idx])
            self.labels = labels[:idx]/255.0
            #print(self.data)
        else:
            #for test
            self.data = tv(data[idx:])
            self.labels = labels[idx:]/255.0
            #print(self.data.shape)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        #img, target = self.data[index], self.data[index]
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
        self.dataset = np.load(self.path)

        #we are gonna to recover the image so the images are labels
        labels = self.dataset['images']
        data = self.dataset['fawkes']

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
    def save_recon(self, recon_img, recon_fawkes, msg = '_'):
        file = self.path[:-4]+'/'+ msg +'.npz'
        np.savez(file, images = recon_img, fawkes = recon_fawkes, labels = self.dataset['labels'])
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
    def __init__(self, path, transform=None, train = True, train_size = 0.8, enhance = True):
        #self.root = os.path.expanduser(path)
        self.path = path
        self.transform = transform

        file = os.path.join(self.path)
        dataset = np.load(file)

        #we are gonna to recover the image so the images are labels
        labels = dataset['images']/255.0
        data = dataset['fawkes']/255.0


        idx = int(data.shape[0] * train_size)

        if train == True:
            self.data = data[:idx]
            self.label = labels[:idx]
            print('add 5x noise')
            noise = self.label - self.data
            self.data = np.clip(self.data + noise * 10, 0, 1) 

        else:
            #for test
            self.data = data[idx:]
            self.label = labels[idx:]
            print('add 5x noise')
            noise = self.label - self.data
            self.data = np.clip(self.data + noise*10, 0, 1) 



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
