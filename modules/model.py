import torch
from .args import device, add_noise, add_jpeg_compression
import torch.nn as nn
from .poincare_concat import PoincareConcatLinear
import geoopt
from geoopt.manifolds.stereographic.math import expmap0, logmap0, project
from .utils import gaussian
import random
# from .jpeg_compress import Jpeg
import torchvision.transforms as transforms

from DiffJPEG.DiffJPEG import DiffJPEG

concat = PoincareConcatLinear(
    in_stacks=2,
    in_dim=3,
    out_dim=6,
    ball=geoopt.PoincareBall(),
    out_split=1
)


# Preparation Network (2 conv layers)
class PrepNetwork(nn.Module):
    def __init__(self):
        super(PrepNetwork, self).__init__()
        self.initialP3 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(50))
        self.initialP4 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(50))
        self.initialP5 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(50))
        self.finalP3 = nn.Sequential(
            nn.Conv2d(150, 1, kernel_size=3, padding=1),
            nn.ReLU())
        self.finalP4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 1, kernel_size=4, padding=2),
            nn.ReLU())
        self.finalP5 = nn.Sequential(
            nn.Conv2d(150, 1, kernel_size=5, padding=2),
            nn.ReLU())

    def forward(self, p):
        p1 = self.initialP3(p)
        p2 = self.initialP4(p)
        p3 = self.initialP5(p)
                
        mid = torch.cat((p1, p2, p3), dim=1)

        p4 = self.finalP3(mid)
        p5 = self.finalP4(mid)
        p6 = self.finalP5(mid)

        out = torch.cat((p4, p5, p6), dim=1)

        return out

# Hiding Network (5 conv layers)
class HidingNetwork(nn.Module):
    def __init__(self):
        super(HidingNetwork, self).__init__()
        self.initialH3 = nn.Sequential(
            nn.Conv2d(6, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(50))
        self.initialH4 = nn.Sequential(
            nn.Conv2d(6, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(50))
        self.initialH5 = nn.Sequential(
            nn.Conv2d(6, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(50))
        self.finalH3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.finalH4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.finalH5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalH = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0))
        
    def forward(self, h):
        h1 = self.initialH3(h)
        h2 = self.initialH4(h)
        h3 = self.initialH5(h)
        
        mid = torch.cat((h1, h2, h3), dim=1)

        h4 = self.finalH3(mid)
        h5 = self.finalH4(mid)
        h6 = self.finalH5(mid)

        mid2 = torch.cat((h4, h5, h6), dim=1)

        out = self.finalH(mid2)
        out_noise = out
        if add_noise: # Set to True to enable training with Gaussian Noise
            r = random.random()
            if r < 0.30:
                out_noise = gaussian(out.data, 0, 0.01)
        return out, out_noise


class EncoderNetwork(nn.Module):
    def __init__(self, concat=concat):
        super(EncoderNetwork, self).__init__()
        self.concat = concat
        self.prep = PrepNetwork()
        self.hiding = HidingNetwork()

    def forward(self, secret, cover):
        s = self.prep(secret)

        # Hyperbolic embedding
        istack = torch.stack((s, cover)).permute(1, 3, 4, 0, 2)
        mid = self.concat.ball.logmap0(self.concat(istack).permute(0, 3, 1, 2))

        # Euclidean
        # mid = torch.cat((s, cover), 1)

        c_prime, c_prime_noise = self.hiding(mid)

        return c_prime, c_prime_noise


# Reveal Network (2 conv layers)
class RevealNetwork(nn.Module):
    def __init__(self):
        super(RevealNetwork, self).__init__()
        self.initialR3 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(50))
        self.initialR4 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(50))
        self.initialR5 = nn.Sequential(
            nn.Conv2d(3, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(50))
        self.finalR3 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=3, padding=1),
            nn.ReLU())
        self.finalR4 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(50, 50, kernel_size=4, padding=2),
            nn.ReLU())
        self.finalR5 = nn.Sequential(
            nn.Conv2d(150, 50, kernel_size=5, padding=2),
            nn.ReLU())
        self.finalR = nn.Sequential(
            nn.Conv2d(150, 3, kernel_size=1, padding=0))

    def forward(self, r):
        r1 = self.initialR3(r)
        r2 = self.initialR4(r)
        r3 = self.initialR5(r)

        mid = torch.cat((r1, r2, r3), dim=1)

        r4 = self.finalR3(mid)
        r5 = self.finalR4(mid)
        r6 = self.finalR5(mid)

        mid2 = torch.cat((r4, r5, r6), dim=1)

        out = self.finalR(mid2)

        return out

# Join three networks in one module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = EncoderNetwork()
        self.reveal = RevealNetwork()
        # self.jpeg = Jpeg()
        self.jpeg = DiffJPEG(height=64, width=64, differentiable=True, quality=80)


    def forward(self, secret, cover):
        c_prime, c_prime_noise = self.encoder(secret, cover)

        if add_jpeg_compression: # Set to True to Enable training with JPEG Compression
            r = random.random()
            if r < 0.30:
                c_prime_noise = self.jpeg(c_prime_noise)
        
        s_prime = self.reveal(c_prime_noise)
        return c_prime, c_prime_noise, s_prime