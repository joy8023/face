import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as keras
from denoise import Denoiser

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

def gen_mask(alpha = 0):
    images, fawkes, labels = load_data('faces/fawkes.npz')
    diff = np.abs(fawkes - images)
    mask = np.zeros(images.shape)
    #print(mask.shape)
    idx = np.where(diff>alpha)
    mask[idx] = 1
    print('percentage of mask:', mask.sum()/2270/112/112/3)

    return mask




def add_mask(images,fawkes):
    print('==========applying mask==========')
    #maskset = np.load('faces/fawkes_mask.npz')
    #img_mask = maskset['images']
    #faw_mask = maskset['fawkes']
    img_mask = gen_mask()
    idx = np.where(img_mask > 0)
    fawkes[idx] = images[idx]
    #fawkes = images
    return fawkes



#get feature of images after denoising directly
#input fawkes.npz
def get_feature(datapath, model_name = 'extractor_0', denoise = False):
    
    model = load_extractor(model_name)
    images, fawkes, labels = load_data(datapath)

    #fawkes = add_mask(images,fawkes)
    #fawkes = images
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



