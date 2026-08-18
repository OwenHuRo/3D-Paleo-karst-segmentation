"""
First revised UPPA model.
This version corrects the channel expansion caused by 3D space-to-depth
downsampling and evaluates SE-reweighted shallow features in skip fusion.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.Modules import EnhancedDownsample3D
class UPPA1(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(UPPA1, self).__init__()
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
        self.conv7 = self.double_conv(nf1*2, nf1)

        self.output_conv = nn.Conv3d(nf1, output_channels, kernel_size=1)

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
        pool1 = self.pool1(conv1)

        conv2 = self.conv2(pool1)
        pool2 = self.pool2(conv2)

        conv3 = self.conv3(pool2)
        pool3 = self.pool3(conv3)

        conv4 = self.conv4(pool3)

        # Upsampling path
        up5 = self.up5(conv4)

        higher2 = self.uplayerinfo2(conv2)

        up5 = torch.cat([higher2, up5, conv3], dim=1)
        conv5 = self.conv5(up5)

        up6 = self.up6(conv5)
        higher1 = self.uplayerinfo1(conv1)
        up6 = torch.cat([higher1, up6, conv2], dim=1)
        conv6 = self.conv6(up6)

        up7 = self.up7(conv6)
        up7 = torch.cat([up7, conv1], dim=1)
        conv7 = self.conv7(up7)

        output = torch.sigmoid(self.output_conv(conv7))
        return output

if __name__ == "__main__":
    input_tensor = torch.rand((1, 1, 128, 128, 128))
    model = UPPA1(input_channels=1, output_channels=1)
    output = model(input_tensor)
    print(output.shape)
