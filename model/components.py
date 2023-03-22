import torch
import torch.nn as nn
import torch.nn.functional as F


class conv_3x3(nn.Module):
    """ (conv ==> [BN] ==> ReLU) * 2 """
    
    def __init__(self, in_channels, out_channels, mid_channels=None, sample=None):
        super(conv_3x3, self).__init__()
        
        if not mid_channels:
            mid_channels = out_channels
        
        self.sample = sample
        self.down = maxPool_2x2()
        self.up = upConv_2x2(in_channels)
        
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(mid_channels), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1), 
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        
        if self.sample == 'up':
            assert skip is not None
            x = self.up(x, skip)
        
        if self.sample == 'down':
            x = self.down(x)
        
        return self.conv_bn_relu(x)


class maxPool_2x2(nn.Module):
    """ for downsampling """
    
    def __init__(self):
        super(maxPool_2x2, self).__init__()
        
        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x):
        return self.maxpool(x)


class upConv_2x2(nn.Module):
    """ for upsampling """
    
    def __init__(self, n_channels):
        super(upConv_2x2, self).__init__()
        
        self.up = nn.ConvTranspose2d(n_channels, n_channels//2, 2, 2)
        # may want to use simple upsample in an if-else block.
        # self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, x, skip):
        x = self.up(x)
        
        # size ---> (B, C, H, W)
        diff_H = skip.size(2) - x.size(2)
        diff_W = skip.size(3) - x.size(3)
        
        x = F.pad(x, (diff_W//2, diff_W - diff_W//2, diff_H//2, diff_H - diff_H//2))
        x = torch.cat([skip, x], dim=1)
        
        return x


class conv_1x1(nn.Module):
    """ map feature vector to the desired number of classes """
    
    def __init__(self, in_channels, num_classes):
        super(conv_1x1, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
    
    def forward(self, x):
        return self.conv(x)
