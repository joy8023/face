#import argparse
#import glob
import numpy as np
import os
#import sys
#from PIL import Image
import tensorflow as tf
import tensorflow.keras as keras
from denoise import Denoiser
from model_resnet import load_model_torch

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


def get_feature_torch(datapath, input_size = [112, 112]):

    model, device = load_model_torch('model/Backbone_ResNet_152_Arcface_Epoch_65.pth')
    batch_size = 128
    images, fawkes, labels = load_data(datapath)

    batch = int(images.shape[0]/batch_size)+1
    image_features = []
    fawkes_features = []

    for i in range(batch):
        if i*batch+batch_size > imgs.shape[0]:
            end = imgs.shape[0]
        else:
            end = i*batch+batch_size
        
        image_b = images[i*batch, end].to(device)
        fawkes_b = fawkes[i*batch, end].to(device)

        image_f = model(image_b).to('cpu').numpy()
        fawkes_f = model(fawkes_b).to('cpu').numpy()

        image_features.append(image_f)
        fawkes_features.append(fawkes_f)

    image_features = np.concatenate(image_features, axis = 0)
    fawkes_features = np.concatenate(fawkes_features, axis = 0)
    
    print(features.shape)

    return image_features, fawkes_features, labels



