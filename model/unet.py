import torch
import torch.nn as nn

from .components import conv_1x1, conv_3x3


class UNet(nn.Module):
    def __init__(self, num_classes=1):
        super(UNet, self).__init__()

        #### DOWN ####
        self.stage1 = conv_3x3(3, 64)
        self.stage2 = conv_3x3(64, 128, sample='down')
        self.stage3 = conv_3x3(128, 256, sample='down')
        self.stage4 = conv_3x3(256, 512, sample='down')
        self.stage5 = conv_3x3(512, 1024, sample='down')

        #### UP ####
        self.stage6 = conv_3x3(1024, 512, sample='up')
        self.stage7 = conv_3x3(512, 256, sample='up')
        self.stage8 = conv_3x3(256, 128, sample='up')
        self.stage9 = conv_3x3(128, 64, sample='up')
        self.stage10 = conv_1x1(64, num_classes)

    def forward(self, x):
        #### DOWN ####
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)
        x5 = self.stage5(x4)
        
        # error in architecture, but still good to go
        
        #### UP ####
        x = self.stage6(x5, x4)
        x = self.stage7(x, x3)
        x = self.stage8(x, x2)
        x = self.stage9(x, x1)
        x = self.stage10(x)

        return x.squeeze(1)


if __name__ == "__main__":
    from torchsummary import summary

    inp = torch.randn(1, 3, 572, 572)
    print(f"Shape of input: {inp.shape}")

    unet = UNet(1)
    out = unet(inp)
    print(f"Shape of output: {out.shape}")
    summary(unet, (3, 572, 572))

    # inp = torch.randn(1, 3, 572, 572)
    # print(f"Shape of input: {inp.shape}")

    # unet = UNet(1)
    # out = unet(inp)
    # print(f"Shape of output: {out.shape}")
