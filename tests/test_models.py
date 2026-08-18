import unittest

import torch

from Models.GSCDUnet import GSCDUnet
from Models.Modules import GSCD, SSS
from Models.UNet import UNet
from Models.UnetPlusPlus import UnetPlusPlus
from Models.UPPA import UPPA
from Models.UPPA1 import UPPA1
from Models.UPPA2 import UPPA2
from Models.UPPA3 import UPPA3
from Utils.utils import CompositeLoss


class ModelSmokeTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12345)
        self.input_tensor = torch.randn(1, 1, 16, 16, 16)

    def test_gscd_unet_shape_and_parameter_count(self):
        model = GSCDUnet(input_channels=1, output_channels=1).eval()
        with torch.inference_mode():
            output = model(self.input_tensor)

        self.assertEqual(tuple(output.shape), tuple(self.input_tensor.shape))
        self.assertEqual(sum(parameter.numel() for parameter in model.parameters()), 7_980_573)
        self.assertTrue(torch.all((output >= 0) & (output <= 1)))

    def test_legacy_model_and_module_names_remain_compatible(self):
        paper_model = GSCDUnet(input_channels=1, output_channels=1)
        legacy_model = UPPA3(input_channels=1, output_channels=1)

        self.assertEqual(list(paper_model.state_dict()), list(legacy_model.state_dict()))
        self.assertIs(SSS, GSCD)

    def test_convolutional_baselines_have_valid_output_shapes(self):
        model_classes = [UNet, UnetPlusPlus, UPPA, UPPA1, UPPA2]
        for model_class in model_classes:
            with self.subTest(model=model_class.__name__):
                model = model_class(input_channels=1, output_channels=1).eval()
                with torch.inference_mode():
                    output = model(self.input_tensor)
                self.assertEqual(tuple(output.shape), tuple(self.input_tensor.shape))

    def test_composite_loss_is_finite(self):
        prediction = torch.sigmoid(torch.randn(1, 1, 8, 8, 8))
        target = torch.zeros_like(prediction)
        target[:, :, 2:6, 2:6, 2:6] = 1

        loss = CompositeLoss()(prediction, target)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))


if __name__ == '__main__':
    unittest.main()
