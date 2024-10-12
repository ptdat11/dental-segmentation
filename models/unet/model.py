import torch.nn as nn
from .parts import *
import config

class Model(nn.Module):
    name = 'unet'
    def __init__(
            self, 
            bn_momentum: float = 0.999,
            bilinear=False):
        super().__init__()
        self.bn_momentum = bn_momentum
        self.bilinear = bilinear

        self.inc = DoubleConv(1, 32, bn_momentum=bn_momentum)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down3 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32, bilinear)

        self.outc = (OutConv(32, config.N_CLASS))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits