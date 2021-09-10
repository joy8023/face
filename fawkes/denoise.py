import argparse
import glob
import numpy as np
import os
import sys
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
        print('=================tv chambolle {} denoising done!============='.format(weight))
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

'''

# rescale_sigma=True required to silence deprecation warnings
_denoise_wavelet = partial(denoise_wavelet, rescale_sigma=True)

sigma = 0.2

images, fawkes, labels = load_data('fawkes.npz')

noisy = fawkes[0]/255.0
original = images[0]/255.0

fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(8, 5),
                       sharex=True, sharey=True)

plt.gray()

# Estimate the average noise standard deviation across color channels.
sigma_est = estimate_sigma(noisy, average_sigmas=True)
#sigma_est = 0.01
# Due to clipping in random_noise, the estimate will be a bit smaller than the
# specified sigma.
print(f'Estimated Gaussian noise standard deviation = {sigma_est}')

parameter_ranges = {'sigma': np.arange(0.001, 0.02, 0.001),
                    'wavelet': ['db1', 'db2'],
                    'convert2ycbcr': [True, False],
                    'multichannel': [True],
                    'method':['BayesShrink', 'VisuShrink']}
calibrated_denoiser = calibrate_denoiser(noisy,
                                         _denoise_wavelet,
                                         denoise_parameters=parameter_ranges)

# Denoised image using calibrated denoiser
calibrated_output = calibrated_denoiser(noisy)

im_bayes = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                           method='BayesShrink', mode='soft',
                           sigma = 0.01, rescale_sigma=True)
im_visushrink = denoise_wavelet(noisy, multichannel=True, convert2ycbcr=True,
                                method='VisuShrink', mode='soft',
                                sigma=sigma_est, rescale_sigma=True)

# Compute PSNR as an indication of image quality
psnr_noisy = peak_signal_noise_ratio(original, noisy)
psnr_bayes = peak_signal_noise_ratio(original, im_bayes)
psnr_visushrink = peak_signal_noise_ratio(original, im_visushrink)
psnr_visushrink2 = peak_signal_noise_ratio(original, calibrated_output)
#psnr_visushrink4 = peak_signal_noise_ratio(original, im_visushrink4)
print(mean_squared_error(original,noisy))
print(mean_squared_error(im_bayes,noisy))
print(mean_squared_error(im_visushrink,noisy))
print(mean_squared_error(original,calibrated_output))
#print(im_bayes)

ax[0, 0].imshow(noisy)
ax[0, 0].axis('off')
ax[0, 0].set_title(f'Noisy\nPSNR={psnr_noisy:0.4g}')
ax[0, 1].imshow(im_bayes)
ax[0, 1].axis('off')
ax[0, 1].set_title(
    f'Wavelet denoising\n(BayesShrink)\nPSNR={psnr_bayes:0.4g}')
ax[0, 2].imshow(im_visushrink)
ax[0, 2].axis('off')
ax[0, 2].set_title(
    'Wavelet denoising\n(VisuShrink, $\\sigma=\\sigma_{est}$)\n'
     'PSNR=%0.4g' % psnr_visushrink)
ax[1, 0].imshow(original)
ax[1, 0].axis('off')
ax[1, 0].set_title('Original')
ax[1, 1].imshow(calibrated_output)
ax[1, 1].axis('off')
ax[1, 1].set_title(
    'Wavelet denoising\n(calibrated_output, $\\sigma=\\sigma_{est}/2$)\n'
     'PSNR=%0.4g' % psnr_visushrink2)
ax[1, 2].imshow((original-calibrated_output)*10)
ax[1, 2].axis('off')
#ax[1, 2].set_title(
#    'Wavelet denoising\n(VisuShrink, $\\sigma=\\sigma_{est}/4$)\n'
#     'PSNR=%0.4g' % psnr_visushrink4)
ax[2,0].imshow((original-noisy)*10)
ax[2,0].axis('off')
ax[2,1].imshow((im_bayes - original)*10)
ax[2,1].axis('off')
ax[2,2].imshow((im_visushrink - original)*10)
fig.tight_layout()

plt.show()
'''

'''
def main(*argv):
    if not argv:
        argv = list(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', '-d', type=str,
                        help='the directory that contains images', default='./faces')

    args = parser.parse_args(argv[1:])



if __name__ == '__main__':
    main(*sys.argv)
'''