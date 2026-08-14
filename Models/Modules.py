import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_spatial_gradients(x: torch.Tensor):
    """
    Compute first-order forward differences along the three spatial axes.
    """
    dx = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
    dy = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
    dz = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
    return dx, dy, dz


class FeatureGradientAttention3D(nn.Module):
    """
    Gradient-enhanced spatial attention for 3D feature maps.
    The module constructs a spatial attention descriptor from three sources:
        1. Channel-wise average features.
        2. Channel-wise maximum features.
        3. The average spatial gradient magnitude across channels.
    A 3D convolution combines these descriptors into a single spatial
    attention map. The resulting attention weights are applied to every
    feature channel at the corresponding voxel location.

    This module extends conventional spatial attention by explicitly
    incorporating local boundary and structural information through spatial
    gradients.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 5, 7)
        padding = kernel_size // 2
        # avg + max + grad_mag
        self.conv = nn.Conv3d(3, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    @staticmethod
    def gradient_magnitude(x: torch.Tensor) -> torch.Tensor:
        dx = torch.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        dy = torch.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        dz = torch.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])

        dx = F.pad(dx, (0, 0, 0, 0, 0, 1))
        dy = F.pad(dy, (0, 0, 0, 1, 0, 0))
        dz = F.pad(dz, (0, 1, 0, 0, 0, 0))
        grad = dx + dy + dz  # [B, C, D, H, W]

        grad = grad.mean(dim=1, keepdim=True)
        return grad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out, _ = x.max(dim=1, keepdim=True)
        grad_mag = self.gradient_magnitude(x)
        attn = torch.cat([avg_out, max_out, grad_mag], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return x * attn

class RegionalConcentrationPrior3D(nn.Module):
    """
    Learnable spatial concentration prior for 3D feature maps.

    The module maintains a trainable voxel-wise prior map with a fixed spatial
    shape. The same prior is broadcast across the batch and channel dimensions
    and multiplied with the input features.

    It can be used when the target structures are more likely to occur in
    certain spatial regions of a volume.
    The prior is not constrained by a sigmoid or other activation and can
    therefore learn arbitrary scaling values during training.
    """
    def __init__(self, D, H, W, init_map: torch.Tensor = None):
        super().__init__()
        if init_map is not None:
            assert init_map.shape == (D, H, W)
            self.prior = nn.Parameter(init_map.clone().float())
        else:
            init = torch.rand(D, H, W) * 0.4 + 0.8
            self.prior = nn.Parameter(init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        prior = self.prior.unsqueeze(0).unsqueeze(0)
        prior = prior.expand(B, C, D, H, W)
        return x * prior


class ChannelShuffleDownsample3D(nn.Module):
    """
    Downsample a 3D feature map using a space-to-depth rearrangement.

    Despite the inherited class name, this operation is more accurately
    described as 3D space-to-depth or voxel unshuffle. It divides the input
    volume into interleaved voxel groups and moves those spatial offsets into
    the channel dimension.

    Unlike pooling, this rearrangement does not discard voxel values. For a
    scale factor s, every spatial dimension is reduced by s while the channel
    count is increased by s^3.
    """
    def __init__(self, scale=2):
        super().__init__()
        self.scale = scale

    def forward(self,x):
        b, c, d, h, w = x.shape
        assert d % self.scale == 0 and h % self.scale == 0 and w % self.scale == 0
        out = torch.cat([
            x[:, :, i::self.scale, j::self.scale, k::self.scale]
            for i in range(self.scale)
            for j in range(self.scale)
            for k in range(self.scale)
        ], dim=1)
        return out


class EnhancedDownsample3D(nn.Module):
    """
    Information-preserving 3D downsampling followed by channel attention.

    The input is first rearranged using 3D space-to-depth, which reduces the
    spatial resolution while moving voxel information into the channel
    dimension. A squeeze-and-excitation block then recalibrates the expanded
    channels according to their global responses.
    """
    def __init__(self, input_channel, output_channel, scale=2, reduction=4):
        super().__init__()
        self.scale = scale
        self.shuffle = ChannelShuffleDownsample3D(self.scale)
        self.input = input_channel
        self.output = output_channel
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(self.input * self.scale**3, (self.input * self.scale**3) // reduction, 1),
            nn.ReLU(),
            nn.Conv3d((self.input * self.scale**3) // reduction, self.input * self.scale**3, 1),
            nn.Sigmoid()
        )
        #self.Compress = nn.Conv3d(self.input * self.scale**3, self.output, kernel_size=1)

    def forward(self, x):
        xs = self.shuffle(x)
        x = xs * self.se(xs)  # SE CAM
        #x = self.Compress(x)
        return x


class SpatialAttention3D(nn.Module):  # SAM
    """
    Channel-pooled spatial attention for 3D feature maps.

    The module computes channel-wise average and maximum projections,
    concatenates them, and uses a 3D convolution to produce a voxel-wise
    attention map.

    This is the 3D equivalent of the spatial attention module commonly used
    in convolutional block attention mechanisms.

    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = kernel_size // 2

        self.conv = nn.Conv3d(2, 1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B, 1, D, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        xa = torch.cat([avg_out, max_out], dim=1)  # (B, 2, D, H, W)
        spatial_attn = self.sigmoid(self.conv(xa))  # (B, 1, D, H, W)

        return x * spatial_attn


class SSS(nn.Module):
    """
    Composite spatial-attention and enhanced-downsampling module.

    The module first applies gradient-enhanced spatial attention to emphasize
    structurally important voxels and boundaries. It then performs
    information-preserving space-to-depth downsampling followed by
    squeeze-and-excitation channel attention.
    """
    def __init__(self, input_channel, output_channel, scale=2, reduction=4, kernel_size=7):
        super().__init__()
        self.Spatial = FeatureGradientAttention3D(kernel_size=kernel_size)
        self.DownSamplingPlusSE = EnhancedDownsample3D(input_channel=input_channel, output_channel=output_channel, scale=scale, reduction=reduction)

    def forward(self, x):
        s1 = self.Spatial(x)
        s23 = self.DownSamplingPlusSE(s1)
        return s23


if __name__ == "__main__":
    input_tensor = torch.rand((1, 4, 128, 128, 128))
    approach = FeatureGradientAttention3D()
    output = approach(input_tensor)
    print(output.shape)