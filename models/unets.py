import torch
import torch.nn as nn
from .blocks import ConvBlock, UpConvBlock, AttentionBlock

class AttentionUNet(nn.Module):
    def __init__(self, in_channels = 3, hidden_channels = 64, out_channels = 1):
        super(AttentionUNet, self).__init__()

        self.maxpool = nn.MaxPool2d(kernel_size = 2, stride = 2)

        self.conv1 = ConvBlock(in_channels = in_channels, out_channels = hidden_channels)
        self.conv2 = ConvBlock(in_channels = hidden_channels, out_channels = hidden_channels * 2)
        self.conv3 = ConvBlock(in_channels = hidden_channels * 2, out_channels = hidden_channels * 4)
        self.conv4 = ConvBlock(in_channels = hidden_channels * 4, out_channels = hidden_channels * 8)
        self.conv5 = ConvBlock(in_channels = hidden_channels * 8, out_channels = hidden_channels * 16)

        self.up5 = UpConvBlock(in_channels = hidden_channels * 16, out_channels = hidden_channels * 8)
        self.att5 = AttentionBlock(f_g = hidden_channels * 8, f_l = hidden_channels * 8, f_int = hidden_channels * 4)
        self.upconv5 = ConvBlock(in_channels = hidden_channels * 16, out_channels = hidden_channels * 8)

        self.up4 = UpConvBlock(in_channels = hidden_channels * 8, out_channels = hidden_channels * 4)
        self.att4 = AttentionBlock(f_g = hidden_channels * 4, f_l = hidden_channels * 4, f_int = hidden_channels * 2)
        self.upconv4 = ConvBlock(in_channels = hidden_channels * 8, out_channels = hidden_channels * 4)

        self.up3 = UpConvBlock(in_channels = hidden_channels * 4, out_channels = hidden_channels * 2)
        self.att3 = AttentionBlock(f_g = hidden_channels * 2, f_l = hidden_channels * 2, f_int = hidden_channels)
        self.upconv3 = ConvBlock(in_channels = hidden_channels * 4, out_channels = hidden_channels * 2)

        self.up2 = UpConvBlock(in_channels = hidden_channels * 2, out_channels = hidden_channels)
        self.att2 = AttentionBlock(f_g = hidden_channels, f_l = hidden_channels, f_int = hidden_channels // 2)
        self.upconv2 = ConvBlock(in_channels = hidden_channels * 2, out_channels = hidden_channels)

        self.out = nn.Conv2d(in_channels = hidden_channels, out_channels = out_channels, kernel_size = 1, stride = 1, padding = 0)

    def forward(self, x):
        # encoder
        x1 = self.conv1(x)
        x2 = self.conv2(self.maxpool(x1))
        x3 = self.conv3(self.maxpool(x2))
        x4 = self.conv4(self.maxpool(x3))
        x5 = self.conv5(self.maxpool(x4))

        # decoder + concat
        d5 = self.up5(x5)
        x4 = self.att5(g = d5, x = x4)
        d5 = self.upconv5(torch.cat([x4, d5], dim = 1))

        d4 = self.up4(d5)
        x3 = self.att4(g = d4, x = x3)
        d4 = self.upconv4(torch.cat([x3, d4], dim = 1))

        d3 = self.up3(d4)
        x2 = self.att3(g = d3, x = x2)
        d3 = self.upconv3(torch.cat([x2, d3], dim = 1))

        d2 = self.up2(d3)
        x1 = self.att2(g = d2, x = x1)
        d2 = self.upconv2(torch.cat([x1, d2], dim = 1))

        # last layer
        out = self.out(d2)

        return out