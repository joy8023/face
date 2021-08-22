import time

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
import torch.nn.functional as F

from model import SegNet
from data import FaceScrub, Celeb
import os, shutil

input_nbr = 3
imsize = 112
batch_size = 64
lr = 0.0001
patience = 50
start_epoch = 0
epochs = 50
print_freq = 10
save_folder = 'models'
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def ensure_folder(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def adjust_learning_rate(optimizer, shrink_factor):
    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

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

def save_checkpoint(epoch, model, optimizer, val_loss, is_best):
    ensure_folder(save_folder)
    state = {'model': model,
             'optimizer': optimizer}
    #filename = '{0}/checkpoint_{1}_{2:.3f}.tar'.format(save_folder, epoch, val_loss)
    #torch.save(state, filename)

    torch.save(model.state_dict(), '{0}/train_{1}_{2:.3f}.pth'.format(save_folder, epoch, val_loss ))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(model.state_dict(), '{0}/best_model.pth'.format(save_folder))

def load_model(model, path):
    try:
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint)
    except:
        print("=> load classifier checkpoint '{}' failed".format(path))
        

def to_image(out):
    ndarr = out.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
    print(ndarr.shape)
    return ndarr

def recon(data_loader, model):
    model.eval()  # eval mode (no dropout or batchnorm)

    # Loss function
    # criterion = nn.MSELoss().to(device)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    cloak_loss = 0
    recover_loss = 0
    losses = ExpoAverageMeter()  # loss (per word decoded)
    recon_img = []

    start = time.time()
    plot = True
    ensure_folder('out')
    msg = 'recon'

    with torch.no_grad():
        # Batches
        for i_batch, (x, y) in enumerate(data_loader):
            # Set device options
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.float)

            recon = model(x)

            loss = torch.sqrt((recon - y).pow(2).mean())
            recover_loss += F.mse_loss(recon, x, reduction='sum').item()
            cloak_loss += F.mse_loss(y, x, reduction='sum').item()

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
            if plot:
                truth = y[0:32]
                inverse = recon[0:32]
                out = torch.cat((inverse, truth))
                for i in range(4):
                    out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
                    out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
                vutils.save_image(out, 'out/recon_{}.png'.format(msg.replace(" ", "")), normalize=False)
                plot = False

            recon_img.append(to_image(recon))

    recon_img = np.concatenate(recon_img, axis = 0)
    print(recon_img.shape)


    recover_loss /= len(data_loader.dataset) * imsize * imsize
    cloak_loss /= len(data_loader.dataset) * imsize * imsize
    print('\n Average MSE loss: recover: {:.6f}, cloak: {:.6f},\n'.format(recover_loss,cloak_loss))

    return recon_img


def main():

    transform = transforms.Compose([transforms.ToTensor()])
    recon_set = Fawkes('./fawkes/faces/fawkes.npz', transform = transform)
    data_loader = DataLoader(recon_set, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

    # Create SegNet model
    label_nbr = 3
    model = SegNet(label_nbr)

    # Use appropriate device
    model = model.to(device)
    recon_img = recon(data_loader, model)
    Fawkes.save_recon(recon_img)


if __name__ == '__main__':
    main()
