"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
#from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
# from PIL import Image
"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
from skimage.transform import resize

class MyDataset(data.Dataset):
    """A template dataset class for you to implement custom datasets."""

    def __init__(self, path, train = True, train_size = 0.8):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        #BaseDataset.__init__(self, opt)
        # get the image paths of your dataset;
        #self.image_paths = []  # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = transforms.Compose([transforms.ToTensor()])
        self.path = path
        #self.transform = transform

        #file = os.path.join(self.path)
        dataset = np.load(self.path)

        #to generate truth from fawkes
        truth = (dataset['images']/255.0)
        fawkes = (dataset['fawkes']/255.0)

        idx = int(truth.shape[0] * train_size)

        if train == True:
            self.truth = truth[:idx]
            self.fawkes = fawkes[:idx]
            #print('add 5x noise')
            #noise = self.label - self.data
            #self.data = np.clip(self.data + noise * 10, 0, 1) 

        else:
            #for test
            self.truth = truth[:idx]
            self.fawkes = fawkes[:idx]
            #print('add 5x noise')
            #noise = self.label - self.data
            #self.data = np.clip(self.data + noise*10, 0, 1) 

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        '''
        path = 'temp'    # needs to be a string
        data_A = None    # needs to be a tensor
        data_B = None    # needs to be a tensor
        return {'data_A': data_A, 'data_B': data_B, 'path': path}
        '''
        path = self.path
        A, B = self.fawkes[index], self.truth[index]
        A = self.transform(resize(A,(128,128)))
        B = self.transform(resize(B,(128,128)))

        return {'A': A, 'B': B, 'A_paths': path}

    def __len__(self):
        """Return the total number of images."""
        return len(self.truth)

    #save original images with reconstructed images
    def save_recon(self, recon, msg = '_'):
        print('======processing recon images=====')
        print('recon.shape:', recon.shape)
        num_img = recon.shape[0]
        images = np.zeros((num_img, 112, 112, 3))

        for i in range(num_img):
            images[i] = resize(recon[i], (112,112))

        print(num_img.shape)
        file = self.path[:-4]+ msg +'recon.npz'
        #print(reconre4567yyyu)
        #print('recon.shape:',recon.shape)
        np.savez(file, images = self.dataset['images'], fawkes = images, labels = self.dataset['labels'])
        print('saved as {}'.format(file))


class MyDataLoader():
    """Wrapper class of Dataset class that performs multi-threaded data loading"""

    def __init__(self, opt):
        """Initialize this class

        Step 1: create a dataset instance given the name [dataset_mode]
        Step 2: create a multi-threaded data loader.
        """
        self.opt = opt
        #dataset_class = find_dataset_using_name(opt.dataset_mode)
        self.dataset = MyDataset(opt.dataroot)
        print("dataset [%s] was created" % type(self.dataset).__name__)
        self.dataloader = data.DataLoader(
            self.dataset,
            batch_size=opt.batch_size,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_threads))

    def load_data(self):
        return self

    def __len__(self):
        """Return the number of data in the dataset"""
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        """Return a batch of data"""
        for i, data in enumerate(self.dataloader):
            if i * self.opt.batch_size >= self.opt.max_dataset_size:
                break
            yield data

    def save_recon(self, recon):
        self.dataset.save_recon(recon)
