"""
This file is a 3D adaptation of the official UCTransNet implementation:

    https://github.com/McGregorWwww/UCTransNet

Reference:
    H. Wang, P. Cao, J. Wang, and O. R. Zaiane,
    "UCTransNet: Rethinking the Skip Connections in U-Net from a
    Channel-wise Perspective with Transformer,"
    Proceedings of the AAAI Conference on Artificial Intelligence, 2022.

Main modifications:
    - Extended the original 2D architecture to 3D volumetric segmentation.
    - Replaced 2D convolution, pooling, patch embedding, and reconstruction
      operations with their 3D counterparts.
    - Replaced Batch Normalization with Group Normalization to support
      batch-size-one training.
    - Adjusted the input channels, image size, and tensor reshaping operations
      for volumetric voxel data.

The core UCTransNet and Channel Transformer design remains based on the
original implementation.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from Models.CTrans import ChannelTransformer
from Utils import Config

def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()

def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU'):
    layers = []
    layers.append(ConvGroupNorm(in_channels, out_channels, activation))

    for _ in range(nb_Conv - 1):
        layers.append(ConvGroupNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)

class ConvGroupNorm(nn.Module):
    def __init__(self, in_channels, out_channels, activation='ReLU'):
        super(ConvGroupNorm, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels,
                              kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(num_groups=int(out_channels/4), num_channels=out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)

class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool3d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class CCA(nn.Module):
    """
    CCA Block
    """
    def __init__(self, F_g, F_x):
        super().__init__()
        self.mlp_x = nn.Sequential(
            Flatten(),
            nn.Linear(F_x, F_x))
        self.mlp_g = nn.Sequential(
            Flatten(),
            nn.Linear(F_g, F_x))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # channel-wise attention
        avg_pool_x = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)), stride=(x.size(2), x.size(3), x.size(4)))
        channel_att_x = self.mlp_x(avg_pool_x)
        avg_pool_g = F.avg_pool3d(g, (g.size(2), g.size(3), g.size(4)), stride=(g.size(2), g.size(3), g.size(4)))
        channel_att_g = self.mlp_g(avg_pool_g)
        channel_att_sum = (channel_att_x + channel_att_g)/2.0

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
        x_after_channel = x * scale
        out = self.relu(x_after_channel)
        return out

class UpBlock_attention(nn.Module):
    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)
        self.coatt = CCA(F_g=in_channels//2, F_x=in_channels//2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)

    def forward(self, x, skip_x):
        up = self.up(x)
        skip_x_att = self.coatt(g=up, x=skip_x)
        x = torch.cat([skip_x_att, up], dim=1)  # dim 1 is the channel dimension
        return self.nConvs(x)

class UCTransNet(nn.Module):
    def __init__(self, config,n_channels=1, n_classes=1,img_size=128,vis=False):
        super().__init__()
        self.vis = vis
        self.n_channels = n_channels
        self.n_classes = n_classes
        in_channels = config.base_channel
        self.inc = ConvGroupNorm(n_channels, in_channels)
        self.down1 = DownBlock(in_channels, in_channels*2, nb_Conv=2)
        self.down2 = DownBlock(in_channels*2, in_channels*4, nb_Conv=2)
        self.down3 = DownBlock(in_channels*4, in_channels*8, nb_Conv=2)
        self.down4 = DownBlock(in_channels*8, in_channels*8, nb_Conv=2)
        self.mtc = ChannelTransformer(config, vis, img_size,
                                     channel_num=[in_channels, in_channels*2, in_channels*4, in_channels*8],
                                     patchSize=config.patch_sizes)
        self.up4 = UpBlock_attention(in_channels*16, in_channels*4, nb_Conv=2)
        self.up3 = UpBlock_attention(in_channels*8, in_channels*2, nb_Conv=2)
        self.up2 = UpBlock_attention(in_channels*4, in_channels, nb_Conv=2)
        self.up1 = UpBlock_attention(in_channels*2, in_channels, nb_Conv=2)
        self.outc = nn.Conv3d(in_channels, n_classes, kernel_size=(1,1,1), stride=(1,1,1))
        self.last_activation = nn.Sigmoid() # if using BCELoss

    def forward(self, x):
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x1,x2,x3,x4,att_weights = self.mtc(x1,x2,x3,x4)
        x = self.up4(x5, x4)
        x = self.up3(x, x3)
        x = self.up2(x, x2)
        x = self.up1(x, x1)
        if self.n_classes == 1:
            logits = self.last_activation(self.outc(x))
        else:
            logits = self.outc(x)
        if self.vis:
            return logits, att_weights
        else:
            return logits



if __name__ == "__main__":
    config = Config.get_CTranS_config()
    model = UCTransNet(config)
    input_tensor = torch.rand((1, 1, 128, 128, 128))
    output = model(input_tensor)
    print(output.shape)
