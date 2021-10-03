import time
import sys
import argparse

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
import torch.nn.functional as F

from model import SegNet, REDNet20, REDNet30
from data import Fawkes
import numpy as np

input_nbr = 3
imsize = 112
batch_size = 128
print_freq = 10
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class ExpoAverageMeter(object):
    # Exponential Weighted Average Meter
    def __init__(self, beta=0.9):
        self.reset()

    def reset(self):
        self.beta = 0.9
        self.val = 0
        self.avg = 0
        self.count = 0

    def update(self, val):
        self.val = val
        self.avg = self.beta * self.avg + (1 - self.beta) * self.val

def load_model(model, path):
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
        return model
    except:
        print("=> load classifier checkpoint '{}' failed".format(path))
        

def to_image(out):
    ndarr = out.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    #print(ndarr.shape)
    return ndarr

def recon(data_loader, model, msg):
    model.eval()  # eval mode (no dropout or batchnorm)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    cloak_loss = 0 #fawkes and images
    recover_loss = 0 #fawkes recon and images
    clean_loss = 0  #image recon and images
    losses = ExpoAverageMeter()  # loss (per word decoded)
    
    recon_img = []
    recon_fawkes = []


    start = time.time()
    plot = True
    #msg = '5xf'

    with torch.no_grad():
        # Batches
        for i_batch, (x, y) in enumerate(data_loader):
            # Set device options
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.float)

            recon = model(x)
            images = model(y)

            loss = torch.sqrt((recon - y).pow(2).mean())
            
            recover_loss += F.mse_loss(recon, y, reduction='sum').item()
            cloak_loss += F.mse_loss(x, y, reduction='sum').item()
            clean_loss += F.mse_loss(images, y, reduction='sum').item()

            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i_batch % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i_batch, len(data_loader),
                                                                      batch_time=batch_time,
                                                                     loss=losses))
            recon_img.append(to_image(images))
            recon_fawkes.append(to_image(recon))

            if plot:
                s = 48
                e = s+16
                fawkes = x[s:e]
                recon = recon[s:e]
                fawkes_diff = torch.abs((fawkes - y[s:e])*3).clamp_(0,1)
                recon_diff = torch.abs((recon - y[s:e])*3).clamp_(0,1)
                out = torch.cat((fawkes, recon, fawkes_diff, recon_diff))

                for i in range(2):
                    out[i * 32:i * 32 + 8] = fawkes[i * 8:i * 8 + 8]
                    out[i * 32 + 8:i * 32 + 16] = recon[i * 8:i * 8 + 8]
                    out[i * 32 + 16:i * 32 + 24] = fawkes_diff[i * 8:i * 8 + 8]
                    out[i * 32 + 24:i * 32 + 32] = recon_diff[i * 8:i * 8 + 8]

                vutils.save_image(out, 'out/recon{}.png'.format(msg.replace(" ", "")), normalize=False)
                plot = False

    recon_img = np.concatenate(recon_img, axis = 0)
    recon_fawkes = np.concatenate(recon_fawkes, axis = 0)
    print(recon_img.shape)

    recover_loss /= len(data_loader.dataset) * imsize * imsize
    cloak_loss /= len(data_loader.dataset) * imsize * imsize
    clean_loss /= len(data_loader.dataset) * imsize * imsize
    print('\n Average MSE loss: recover: {:.6f}, cloak: {:.6f}, clean: {:.6f},\n'.format(recover_loss,cloak_loss,clean_loss))

    return recon_img, recon_fawkes


def main(*argv):

    if not argv:
        argv = list(sys.argv)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str,
                        help='the path of model', default='models/best_model_fawkes.pth')
    parser.add_argument('--data', '-d', type=str,
                        help='the path of data set', default= 'faces/fawkes.npz')
    parser.add_argument('--msg', '-msg', type=str,
                        help='msg for saving files', default='_')
    args = parser.parse_args(argv[1:])


    transform = transforms.Compose([transforms.ToTensor()])
    data_set = Fawkes(args.data, transform = transform)
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False)

    path = args.model
    # Create SegNet model
    #label_nbr = 3
    #model = SegNet(label_nbr)
    model = REDNet30()
    model = load_model(model, path)


    # Use appropriate device
    #msg = '_'
    model = model.to(device)
    recon_img, recon_fawkes = recon(data_loader, model, args.msg)
    data_set.save_recon(recon_img, recon_fawkes, args.msg)


if __name__ == '__main__':
    main(*sys.argv)
