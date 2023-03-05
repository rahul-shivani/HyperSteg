from pathlib import Path

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

# HELPERS

import argparse, torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from pathlib import Path

class One_Hot(nn.Module):
    # from :
    # https://lirnli.wordpress.com/2017/09/03/one-hot-encoding-in-pytorch/
    def __init__(self, depth):
        super(One_Hot,self).__init__()
        self.depth = depth
        self.ones = torch.sparse.torch.eye(depth)
    def forward(self, X_in):
        X_in = X_in.long()
        return Variable(self.ones.index_select(0,X_in.data))
    def __repr__(self):
        return self.__class__.__name__ + "({})".format(self.depth)


def cuda(tensor,is_cuda):
    if is_cuda : return tensor.cuda()
    else : return tensor

def str2bool(v):
    # codes from : https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse

    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def rm_dir(dir_path, silent=True):
    p = Path(dir_path).resolve()
    if (not p.is_file()) and (not p.is_dir()) :
        print('It is not path for file nor directory :',p)
        return

    paths = list(p.iterdir())
    if (len(paths) == 0) and p.is_dir() :
        p.rmdir()
        if not silent : print('removed empty dir :',p)

    else :
        for path in paths :
            if path.is_file() :
                path.unlink()
                if not silent : print('removed file :',path)
            else:
                rm_dir(path)
        p.rmdir()
        if not silent : print('removed empty dir :',p)

def where(cond, x, y):
    """
    code from :
        https://discuss.pytorch.org/t/how-can-i-do-the-operation-the-same-as-np-where/1329/8
    """
    cond = cond.float()
    return (cond*x) + ((1-cond)*y)


class Attack(object):
    def __init__(self, net, criterion):
        self.net = net
        self.criterion = criterion

    def fgsm(self, x, x_c, y, y_c, targeted=False, eps=0.03, x_val_min=-1, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        x_c_adv = Variable(x_c.data, requires_grad=True)
        h_c_adv, _, h_adv = self.net(x_adv, x_c_adv)

        if targeted:
            cost, _, _ = self.criterion(h_adv, h_c_adv, y, y_c, 1.0)
        else:
            cost, _, _ = self.criterion(h_adv, h_c_adv, y, y_c, 1.0)
            cost = -cost

        self.net.zero_grad()
        if x_adv.grad is not None:
            x_adv.grad.data.fill_(0)
        if x_c_adv.grad is not None:
            x_c_adv.grad.data.fill_(0)
        cost.backward()

        x_adv.grad.sign_()
        x_adv = x_adv - eps*x_adv.grad
        x_adv = torch.clamp(x_adv, x_val_min, x_val_max)

        x_c_adv.grad.sign_()
        x_c_adv = x_c_adv - eps*x_c_adv.grad
        x_c_adv = torch.clamp(x_c_adv, x_val_min, x_val_max)


        h_c, _, h = self.net(x, x_c)
        h_c_adv, _, h_adv = self.net(x_adv, x_c_adv)

        return x_adv, x_c_adv, h_adv, h_c_adv, h, h_c

    def i_fgsm(self, x, x_c, y, y_c, targeted=False, eps=0.03, alpha=1, iteration=1, x_val_min=-1, x_val_max=1):
        x_adv = Variable(x.data, requires_grad=True)
        x_c_adv = Variable(x_c.data, requires_grad=True)

        for i in range(iteration):
            h_c_adv, _,  h_adv = self.net(x_adv, x_c_adv)
            if targeted:
                cost, _, _ = self.criterion(h_adv, h_c_adv, y, y_c, 1.0)
            else:
                cost, _, _ = self.criterion(h_adv, h_c_adv, y, y_c, 1.0)
                cost = -cost

            self.net.zero_grad()
            if x_adv.grad is not None:
                x_adv.grad.data.fill_(0)
            if x_c_adv.grad is not None:
                x_c_adv.grad.data.fill_(0)
            cost.backward()

            x_adv.grad.sign_()
            x_adv = x_adv - alpha*x_adv.grad
            x_adv = where(x_adv > x+eps, x+eps, x_adv)
            x_adv = where(x_adv < x-eps, x-eps, x_adv)
            x_adv = torch.clamp(x_adv, x_val_min, x_val_max)
            x_adv = Variable(x_adv.data, requires_grad=True)

            x_c_adv.grad.sign_()
            x_c_adv = x_c_adv - alpha*x_c_adv.grad
            x_c_adv = where(x_c_adv > x_c+eps, x_c+eps, x_c_adv)
            x_c_adv = where(x_c_adv < x_c-eps, x_c-eps, x_c_adv)
            x_c_adv = torch.clamp(x_c_adv, x_val_min, x_val_max)
            x_c_adv = Variable(x_c_adv.data, requires_grad=True)

        h_c, _, h = self.net(x, x_c)
        h_c_adv, _, h_adv = self.net(x_adv, x_c_adv)

        return x_adv, x_c_adv, h_adv,  h_c_adv, h, h_c

    def universal(self, args):
        self.set_mode('eval')

        init = False

        correct = 0
        cost = 0
        total = 0

        data_loader = self.data_loader['test']
        for e in range(100000):
            for batch_idx, (images, labels) in enumerate(data_loader):

                x = Variable(cuda(images, self.cuda))
                y = Variable(cuda(labels, self.cuda))

                if not init:
                    sz = x.size()[1:]
                    r = torch.zeros(sz)
                    r = Variable(cuda(r, self.cuda), requires_grad=True)
                    init = True

                logit = self.net(x+r)
                p_ygx = F.softmax(logit, dim=1)
                H_ygx = (-p_ygx*torch.log(self.eps+p_ygx)).sum(1).mean(0)
                prediction_cost = H_ygx
                #prediction_cost = F.cross_entropy(logit,y)
                #perceptual_cost = -F.l1_loss(x+r,x)
                #perceptual_cost = -F.mse_loss(x+r,x)
                #perceptual_cost = -F.mse_loss(x+r,x) -r.norm()
                perceptual_cost = -F.mse_loss(x+r, x) -F.relu(r.norm()-5)
                #perceptual_cost = -F.relu(r.norm()-5.)
                #if perceptual_cost.data[0] < 10: perceptual_cost.data.fill_(0)
                cost = prediction_cost + perceptual_cost
                #cost = prediction_cost

                self.net.zero_grad()
                if r.grad:
                    r.grad.fill_(0)
                cost.backward()

                #r = r + args.eps*r.grad.sign()
                r = r + r.grad*1e-1
                r = Variable(cuda(r.data, self.cuda), requires_grad=True)



                prediction = logit.max(1)[1]
                correct = torch.eq(prediction, y).float().mean().data[0]
                if batch_idx % 100 == 0:
                    if self.visdom:
                        self.vf.imshow_multi(x.add(r).data)
                        #self.vf.imshow_multi(r.unsqueeze(0).data,factor=4)
                    print(correct*100, prediction_cost.data[0], perceptual_cost.data[0],\
                            r.norm().data[0])

        self.set_mode('train')