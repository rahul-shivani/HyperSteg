import torch
import matplotlib.pyplot as plt
from torch.autograd import Variable
from .args import std, mean, device
import numpy as np
import matplotlib.image

def reveal_loss(S_prime, S):
    ''' Calculates reveal loss specified on the paper.'''

    loss_secret = torch.nn.functional.mse_loss(S_prime, S)
    return loss_secret

def encoder_loss(C_prime, C):
    ''' Calculates encoder loss specified on the paper.'''

    loss_cover = torch.nn.functional.mse_loss(C_prime, C)
    return loss_cover


def customized_loss(S_prime, C_prime, S, C, B):
    ''' Calculates loss specified on the paper.'''
    
    loss_cover = encoder_loss(C_prime, C)
    loss_secret = reveal_loss(S_prime, S)
    loss_all = loss_cover + B*loss_secret
    return loss_all, loss_cover, loss_secret

def denormalize(image, std, mean):
    ''' Denormalizes a tensor of images.'''

    for t in range(3):
        image[t, :, :] = (image[t, :, :] * std[t]) + mean[t]
    return image

def imshow(img, idx, learning_rate, beta):
    '''Prints out an image given in tensor format.'''
    
    img = denormalize(img, std, mean)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title('Example '+str(idx)+', lr='+str(learning_rate)+', B='+str(beta))
    plt.show()
    # return

def imsave(img, path):
    '''Prints out an image given in tensor format.'''
    
    img = denormalize(img, std, mean)
    npimg = img.numpy()
    npimg = np.clip(npimg, a_min = 0, a_max = 1)
    matplotlib.image.imsave(path, np.transpose(npimg, (1, 2, 0)))


def gaussian(tensor, mean=0, stddev=0.1):
    '''Adds random noise to a tensor.'''
    
    noise = torch.nn.init.normal_(torch.Tensor(tensor.size()), mean, stddev).to(device)
    return Variable(tensor + noise)