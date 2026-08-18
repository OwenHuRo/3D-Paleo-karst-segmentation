"""
GSCD-Unet for 3D paleokarst cave semantic segmentation.

This is the public model name used in the accompanying manuscript. The
original experimental code called the same architecture ``UPPA3``; subclassing
it keeps existing state-dict checkpoints and the legacy import path compatible.
"""

from Models.UPPA3 import UPPA3


class GSCDUnet(UPPA3):
    """Paper-facing name of the final UPPA3 architecture."""

    pass


if __name__ == "__main__":
    import torch

    input_tensor = torch.rand((1, 1, 128, 128, 128))
    model = GSCDUnet(input_channels=1, output_channels=1)
    output = model(input_tensor)
    print(output.shape)
