import torch
from torchvision import transforms
import torchvision.utils as vutils

import numpy as np

def load_data(path1, path2):
    data1 = np.load(path1)
    images = data1['images']
    fawkes = data1['fawkes']

    #recon data
    data2 = np.load(path2)
    images_recon = data2['images']
    fawkes_recon = data2['fawkes']

    s = 48
    e = s + 8

    out = np.concatenate((images[s:e], fawkes[s:e]), axis = 0)
    out = np.concatenate((out, images_recon[s:e]), axis = 0)
    out = np.concatenate((out, fawkes_recon[s:e]), axis = 0)
    out = np.concatenate((out, np.abs(fawkes[s:e]-images[s:e])), axis = 0)
    out = np.concatenate((out, np.abs(images_recon[s:e]-images[s:e])), axis = 0)
    out = np.concatenate((out, np.abs(fawkes_recon[s:e]-images[s:e])), axis = 0)
    out = np.concatenate((out, images[s:e]), axis = 0)
    print(out.shape)
    out = np.transpose(out, (0,3,1,2))/255.0
    out = torch.Tensor(out)

    vutils.save_image(out, 'faces/recon/{}.png'.format(msg.replace(" ", "")), normalize=False)

path1 = 'faces/fawkes.npz'
path2 = 'faces/fawkes/rn305xf.npz'
msg = '5x'
load_data(path1,path2)