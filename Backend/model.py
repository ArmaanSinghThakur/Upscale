import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp

# Constants from your snippet
NUM_FEATURES = 64
GROWTH_CHANNEL = 32
NUM_RRDB_BLOCKS = 6  # Matched to your latest snippet

class ResidualDenseBlockLight(nn.Module):
    def __init__(self, nf=NUM_FEATURES, gc=GROWTH_CHANNEL):
        super().__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + 2*gc, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + 3*gc, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + 4*gc, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDBLight(nn.Module):
    def __init__(self, nf=NUM_FEATURES, gc=GROWTH_CHANNEL):
        super().__init__()
        self.blocks = nn.Sequential(
            ResidualDenseBlockLight(nf, gc),
            ResidualDenseBlockLight(nf, gc),
            ResidualDenseBlockLight(nf, gc)
        )

    def forward(self, x):
        return self.blocks(x) * 0.2 + x

class RRDBNetLight(nn.Module):
    def __init__(self, in_nc=3, out_nc=3, nf=NUM_FEATURES, nb=NUM_RRDB_BLOCKS, gc=GROWTH_CHANNEL):
        super().__init__()
        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1)
        self.blocks = nn.ModuleList([RRDBLight(nf, gc) for _ in range(nb)])
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        
        # keep convs simple — interpolation does spatial upscaling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1)
        self.hr_conv = nn.Conv2d(nf, nf, 3, 1, 1)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = fea
        
        # Using standard loop instead of checkpointing for inference stability
        # (Checkpointing is mainly for training memory efficiency)
        for blk in self.blocks:
            trunk = blk(trunk)
            
        fea = fea + self.trunk_conv(trunk)
        
        # upsample x2 twice (x4)
        fea = self.lrelu(self.upconv1(F.interpolate(fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(fea, scale_factor=2, mode='nearest')))
        
        out = self.conv_last(self.lrelu(self.hr_conv(fea)))
        
        return out