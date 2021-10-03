import time

import torch
import torch.optim as optim
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils

from model import SegNet, REDNet20, REDNet30
from data import *
import os, shutil, sys
from resnet import load_resnet

input_nbr = 3
imsize = 112
batch_size = 32
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

    #torch.save(model.state_dict(), '{0}/train_{1}_{2:.3f}.pth'.format(save_folder, epoch, val_loss ))
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(model.state_dict(), '{0}/best_rn30_mask.pth'.format(save_folder))


def train(epoch, train_loader, model, optimizer):
    # Ensure dropout layers are in train mode
    model.train()
    # Loss function
    # criterion = nn.MSELoss().to(device)
    msg = 'train'
    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)

    start = time.time()
    plot = True
    # Batches
    for i_batch, (x, y) in enumerate(train_loader):
    
        # Set device options
        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.float)

        # Zero gradients
        optimizer.zero_grad()

        y_hat = model(x)
        # print('y_hat.size(): ' + str(y_hat.size())) # [32, 3, 224, 224]

        loss = torch.sqrt((y_hat - y).pow(2).mean())
        loss.backward()

        # optimizer.step(closure)
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        if plot:
            truth = y[0:32]
            inverse = y_hat[0:32]
            out = torch.cat((inverse, truth))
            for i in range(4):
                out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
                out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
            vutils.save_image(out, 'out/recon_{}_{}.png'.format(msg.replace(" ", ""), epoch), normalize=False)
            plot = False

        # Print status
        if i_batch % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i_batch, len(train_loader),
                                                                  batch_time=batch_time,
                                                                  loss=losses))
#train model with feature loss
def train2(resnet, epoch, train_loader, model, optimizer):
    # Ensure dropout layers are in train mode
    model.train()

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)

    a = 20
    start = time.time()

    # Batches
    for i_batch, (x, y) in enumerate(train_loader):
        # Set device options
        x = x.to(device, dtype=torch.float)
        y = y.to(device, dtype=torch.float)

        # Zero gradients
        optimizer.zero_grad()

        y_hat = model(x)
        feature = resnet(y)
        feature_hat = resnet(y_hat)

        feature_loss = nn.functional.l1_loss(feature, feature_hat)
        image_loss = torch.sqrt((y_hat - y).pow(2).mean())

        loss = image_loss + a * feature_loss
        #print(image_loss.item(),feature_loss.item(),loss.item())
        loss.backward()

        # optimizer.step(closure)
        optimizer.step()

        # Keep track of metrics
        losses.update(loss.item())
        batch_time.update(time.time() - start)

        start = time.time()

        # Print status
        if i_batch % print_freq == 0:
            print('Epoch: [{}][{}/{}]\t'
                  'image {:.6f} feature {:.6f} \t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(epoch, i_batch, len(train_loader),
                                                                  image_loss.item(),feature_loss.item(),
                                                                  loss=losses))

def valid2(resnet, val_loader, model, epoch):
    model.eval()  # eval mode (no dropout or batchnorm)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)

    start = time.time()
    plot = True
    ensure_folder('out')
    msg = 'rn30_10x_f'
    a = 20
    with torch.no_grad():
        # Batches
        for i_batch, (x, y) in enumerate(val_loader):
            # Set device options
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.float)

            y_hat = model(x)

            feature = resnet(y)
            feature_hat = resnet(y_hat)

            feature_loss = nn.functional.l1_loss(feature, feature_hat)
            image_loss = torch.sqrt((y_hat - y).pow(2).mean())

            loss = image_loss + a * feature_loss
            #loss = torch.sqrt((y_hat - y).pow(2).mean())

            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i_batch % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {2:.6f} ({3:.6f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i_batch, len(val_loader),
                                                                      image_loss.item(),feature_loss.item(),
                                                                      loss=losses))
            if plot:
                truth = y[0:32]
                inverse = y_hat[0:32]
                out = torch.cat((inverse, truth))
                for i in range(4):
                    out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
                    out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
                vutils.save_image(out, 'out/recon_{}_{}.png'.format(msg.replace(" ", ""), epoch), normalize=False)
                plot = False
                #print(y)

    return losses.avg

def valid(val_loader, model, epoch):
    model.eval()  # eval mode (no dropout or batchnorm)

    batch_time = ExpoAverageMeter()  # forward prop. + back prop. time
    losses = ExpoAverageMeter()  # loss (per word decoded)

    start = time.time()
    plot = True
    ensure_folder('out')
    msg = 'rn30'

    with torch.no_grad():
        # Batches
        for i_batch, (x, y) in enumerate(val_loader):
            # Set device options
            x = x.to(device, dtype=torch.float)
            y = y.to(device, dtype=torch.float)

            y_hat = model(x)

            loss = torch.sqrt((y_hat - y).pow(2).mean())

            # Keep track of metrics
            losses.update(loss.item())
            batch_time.update(time.time() - start)

            start = time.time()

            # Print status
            if i_batch % print_freq == 0:
                print('Validation: [{0}/{1}]\t'
                      'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(i_batch, len(val_loader),
                                                                      batch_time=batch_time,
                                                                      loss=losses))
            if plot:
                truth = y[0:32]
                inverse = y_hat[0:32]
                out = torch.cat((inverse, truth))
                for i in range(4):
                    out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
                    out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
                vutils.save_image(out, 'out/recon_{}_{}.png'.format(msg.replace(" ", ""), epoch), normalize=False)
                plot = False
                #print(y)

    return losses.avg

#train inversion model
def train_inv(epoch, data_loader, model, optimizer):

    model.train()

    for batch_idx, (feature, image) in enumerate(data_loader):
        feature, image = feature.to(device), image.to(device)
        optimizer.zero_grad()

        recon = model(feature)
        loss = F.mse_loss(recon, image)
        loss.backward()
        optimizer.step()

        if batch_idx % print_freq == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f}'.format( epoch, batch_idx * len(data),
                                                                  len(data_loader.dataset), loss.item()))

def test_inv(epoch, data_loader, model, msg):

    mse_loss = 0
    plot = True

    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)

            recon = model(data)
            mse_loss += F.mse_loss(recon, data, reduction='sum').item()

            if plot:
                truth = target[0:32]
                inverse = recon[0:32]
                out = torch.cat((inverse, truth))
                for i in range(4):
                    out[i * 16:i * 16 + 8] = inverse[i * 8:i * 8 + 8]
                    out[i * 16 + 8:i * 16 + 16] = truth[i * 8:i * 8 + 8]
                vutils.save_image(out, 'out/recon_{}_{}.png'.format(msg.replace(" ", ""), epoch), normalize=False)
                plot = False

    mse_loss /= len(data_loader.dataset) * 112 * 112
    print('\nTest inversion model on {} set: Average MSE loss: {:.6f}\n'.format(msg, mse_loss))
    return mse_loss

def main():

    transform = transforms.Compose([transforms.ToTensor()])
    #train_set = FaceScrub('./face.npz', transform=transform)
    #test_set = FaceScrub('./face_test.npz', transform=transform)

    #train_set = Celeb('./data/celeba_3w.npy', transform = transform)
    #train_set = Celeb('./data/celeba_1w.npy', transform = transform)
    #test_set = Celeb('./data/celeba_1w.npy', transform = transform, train = False)
    train_set = CelebMask('./data/celebwmask1w.npz', transform = transform)
    test_set = CelebMask('./data/celebwmask1w.npz', transform = transform, train = False)

    #train_set = Fawkes_train('./fawkes/celeba_1w_fawkes.npz', transform = transform)
    #test_set = Fawkes_train('./fawkes/celeba_1w_fawkes.npz', transform = transform, train = False)


    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=True)

    # Create SegNet model
    label_nbr = 3
    #model = SegNet(label_nbr)
    
    # Use appropriate device

    #model = model.to(device)
    #model = SegNet(label_nbr).to(device)
    #optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    # optimizer = optim.LBFGS(model.parameters(), lr=0.8)

    #model and optimizer for rednet
    model = REDNet30().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
          
    #model and opt for inversion model
    #model = Inversion().to(device)
    #optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999), amsgrad=True)

    best_loss = 100000
    epochs_since_improvement = 0
    
    #resnet, _ = load_resnet('fawkes/model/Backbone_ResNet_152_Arcface_Epoch_65.pth')
    #resnet.eval()
    
    # Epochs
    for epoch in range(start_epoch, epochs):
        # Decay learning rate if there is no improvement for 8 consecutive epochs, and terminate training after 20
        if epochs_since_improvement == 20:
            break
        if epochs_since_improvement > 0 and epochs_since_improvement % 8 == 0:
            adjust_learning_rate(optimizer, 0.8)

        # One epoch's training
        print('training')
        train(epoch, train_loader, model, optimizer)
        #train2(resnet, epoch, train_loader, model, optimizer)

        # One epoch's validation
        val_loss = valid(val_loader, model, epoch)
        #val_loss = valid2(resnet, val_loader, model, epoch)
        print('\n * LOSS - {loss:.3f}\n'.format(loss=val_loss))

        # Check if there was an improvement
        is_best = val_loss < best_loss
        best_loss = min(best_loss, val_loss)

        if not is_best:
            epochs_since_improvement += 1
            print("\nEpochs since last improvement: %d\n" % (epochs_since_improvement,))
        else:
            epochs_since_improvement = 0

        # Save checkpoint
        save_checkpoint(epoch, model, optimizer, val_loss, is_best)


if __name__ == '__main__':
    main()
