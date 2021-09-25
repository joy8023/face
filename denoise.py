import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
from functools import partial
from skimage.restoration import (denoise_wavelet, estimate_sigma, 
                                calibrate_denoiser, denoise_nl_means,
                                denoise_tv_chambolle, denoise_bilateral)
from skimage.util import random_noise

def load_data(datapath):
    data = np.load(datapath)
    images = data['images']
    fawkes = data['fawkes']
    labels = data['labels']
    return images, fawkes, labels

def to_array(l):
    a = np.array(l)
    print('after denoising:', a.shape)
    return a


class Denoiser(object):
    def __init__(self, data):
        self.data = data

    #calibrate wavelet denoiser
    def cal_wave(self):
        _denoise_wavelet = partial(denoise_wavelet, rescale_sigma=True)
        parameter_ranges = {'sigma': np.arange(0.001, 0.02, 0.001),
                    'wavelet': ['db1', 'db2'],
                    'convert2ycbcr': [True, False],
                    'multichannel': [True],
                    'method':['BayesShrink', 'VisuShrink']}
        #for noisy in self.data:
        calibrated_denoiser = calibrate_denoiser(noisy,
                                         _denoise_wavelet,
                                         denoise_parameters=parameter_ranges)

        data = []
        for noisy in self.data:
            output = calibrated_denoiser(noisy)
            data.append(output)
        data = to_array(data)

        data = np.uint8(data * 255 + 0.5)
        print(data)
        print('=================cal wavelet denoising done!=============')
        return data

    def wavelet(self, sigma = 0.05):

        data = []
        for noisy in self.data:
            output = denoise_wavelet(noisy, multichannel = True, convert2ycbcr=True,
                            method='BayesShrink', mode='soft',sigma=sigma,
                                rescale_sigma=True)
            data.append(output)
        data = to_array(data)

        data = np.uint8(data * 255 + 0.5)
        #print(data)
        print('=================cal wavelet denoising done!=============')
        return data

    def nl_mean(self, sigma = 0.05):
        data = []

        for noisy in self.data:
            #noisy = random_noise(noisy, var=sigma**2)
            patch_kw = dict(patch_size=5,      # 5x5 patches
                patch_distance=6,  # 13x13 search area
                multichannel = True)

            # slow algorithm
            denoise = denoise_nl_means(noisy, h=0.8 * sigma, fast_mode=True,
                           sigma = sigma, **patch_kw)
            data.append(denoise)

        data = to_array(data)
        data = np.uint8(data * 255 + 0.5)
        print('=================nl_mean denoising done!=============')
        return data

    def tv(self, weight = 0.3):
        data = []

        for noisy in self.data:
            #noisy = random_noise(noisy, var=sigma**2)
            # slow algorithm
            denoise = denoise_tv_chambolle(noisy, weight=weight, multichannel = True)
            data.append(denoise)

        data = to_array(data)
        data = np.uint8(data * 255 + 0.5)
        print('=================tv chambolle (weight:{}) denoising done!============='.format(weight))
        return data

    def bilateral(self):
        data = []

        for noisy in self.data:
            #noisy = random_noise(noisy, var=sigma**2)
            # slow algorithm
            denoise = denoise_bilateral(noisy, sigma_color=0.02, 
                    sigma_spatial=10, multichannel = True)
            data.append(denoise)

        data = to_array(data)
        data = np.uint8(data * 255 + 0.5)
        print('=================bilateral denoising done!=============')
        return data
