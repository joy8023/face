import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from denoise import Denoiser
#from model_resnet import load_model_torch
#import torch
#from torchvision import transforms
#from torch.utils.data import DataLoader
#from torch.utils.data import Dataset
'''
'''
def l2_norm(x, axis=1):
    """l2 norm"""
    norm = tf.norm(x, axis=axis, keepdims=True)
    output = x / norm
    return output

class Extractor(object):
    def __init__(self, model):
        self.model = model

    def predict(self, imgs ):
        imgs = imgs / 255.0
        embeds = l2_norm(self.model.predict(imgs,batch_size = 64))
        return embeds

    def __call__(self, x):
        print('flag')
        return self.predict(x,batch_size = 32)

def load_extractor(name):

    model = keras.models.load_model("model/{}.h5".format(name))
    model = Extractor(model)

    return model

def load_data(datapath):
    data = np.load(datapath)
    images = data['images']
    fawkes = data['fawkes']
    labels = data['labels']
    return images, fawkes, labels

#get feature of images after denoising directly
#input fawkes.npz
def get_feature(datapath, model_name = 'extractor_0', denoise = False):
    
    model = load_extractor(model_name)
    images, fawkes, labels = load_data(datapath)
    
    if denoise:
        denoiser = Denoiser(fawkes)
        fawkes = denoiser.tv()
        #fawkes = denoiser.nl_mean()
        #fawkes = 0.4* fawkes2 + 0.6*fawkes1
        #fawkes = denoiser.bilateral()
        np.savez(datapath[:-4]+'_tv.npz', images = images, fawkes = fawkes, labels = labels)
    image_features = model.predict(images)
    fawkes_features = model.predict(fawkes)
    print('successfully load features')
    return np.array(image_features), np.array(fawkes_features), labels



