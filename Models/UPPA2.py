"""
Spatial-attention experiment derived from UPPA1.

This version evaluates spatially attended encoder features and an additional
shallow-feature fusion path. Several tested components are retained in the
file.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Modules import EnhancedDownsample3D, SpatialAttention3D, RegionalConcentrationPrior3D


class UPPA2(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UPPA2, self).__init__()
        nf1 = 32
        nf2 = 64
        nf3 = 128
        nf4 = 256

        # Downsampling path
        self.conv1 = self.double_conv(input_channels, nf1)
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.conv2 = self.double_conv(nf1, nf2)
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.conv3 = self.double_conv(nf2, nf3)
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        self.conv4 = self.double_conv(nf3, nf4)

        self.uplayerinfo1 = EnhancedDownsample3D(nf1, nf1)

        self.uplayerinfo2 = EnhancedDownsample3D(nf2, nf2)

        # Upsampling path
        self.up5 = nn.ConvTranspose3d(nf4, nf3, kernel_size=2, stride=2)
        self.conv5 = self.double_conv(nf3*2+nf2*8, nf3)

        self.up6 = nn.ConvTranspose3d(nf3, nf2, kernel_size=2, stride=2)

        self.conv6 = self.double_conv(nf2*2+nf1*8, nf2)

        self.up7 = nn.ConvTranspose3d(nf2, nf1, kernel_size=2, stride=2)
        self.conv7 = self.double_conv(nf1*3, nf1)

        self.output_conv = nn.Conv3d(nf1, output_channels, kernel_size=1)
        # skip
        self.space_cul = SpatialAttention3D()
        self.up_mid2 = nn.ConvTranspose3d(nf2, nf1, kernel_size=2, stride=2)
        self.mid12 = self.double_conv(nf1+nf1, nf1)
        self.mid1_down = EnhancedDownsample3D(nf1, nf1*8)
        self.mid2_down = EnhancedDownsample3D(nf2, nf2*8)

    def double_conv(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Downsampling path
        conv1 = self.conv1(x)
        mid1 = self.space_cul(conv1)
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        mid2 = self.space_cul(conv2)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)

        # Upsampling path
        up5 = self.up5(conv4)

        mid2_down = self.mid2_down(mid2)

        up5 = torch.cat([mid2_down, up5, conv3], dim=1)
        conv5 = self.conv5(up5)

        up6 = self.up6(conv5)

        mid1_down = self.mid1_down(mid1)

        up6 = torch.cat([mid1_down, up6, mid2], dim=1)
        conv6 = self.conv6(up6)

        up_mid2 = self.up_mid2(mid2)

        mid12 = torch.cat([mid1, up_mid2], dim=1)

        mid12 = self.mid12(mid12)

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, mid12, conv1], dim=1)
        conv7 = self.conv7(up7)

        output = torch.sigmoid(self.output_conv(conv7))
        return output

model = UPPA2(input_channels=1, output_channels=1)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

if __name__ == "__main__":
    input_tensor = torch.rand((1, 1, 128, 128, 128))
    output = model(input_tensor)
    print(output.shape)
