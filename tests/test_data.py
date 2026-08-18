import tempfile
import unittest
from pathlib import Path

import numpy as np

from Utils.utils import DataGenerator


class DataGeneratorSmokeTests(unittest.TestCase):
    def test_memory_mapped_volume_loading(self):
        volume_shape = (64, 64, 64)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / 'GroundTruth'
            label_dir = root / 'Label'
            data_dir.mkdir()
            label_dir.mkdir()

            data = np.linspace(-1, 1, num=np.prod(volume_shape), dtype=np.float32).reshape(volume_shape)
            label = np.zeros(volume_shape, dtype=np.float32)
            label[20:40, 20:40, 20:40] = 1
            data.tofile(data_dir / 'synthetic_seismic_final_1.dat')
            label.tofile(label_dir / 'synthetic_seismic_final_1.dat')

            dataset = DataGenerator(
                dpath=data_dir,
                fpath=label_dir,
                data_IDs=[1],
                dim=volume_shape,
                Enhance=False,
                SmallScale=True
            )
            input_tensor, label_tensor = dataset[0]

            self.assertEqual(tuple(input_tensor.shape), (1, 64, 64, 64))
            self.assertEqual(tuple(label_tensor.shape), (1, 64, 64, 64))
            self.assertAlmostEqual(float(input_tensor.mean()), 0.0, places=5)
            self.assertEqual(float(label_tensor.max()), 1.0)


if __name__ == '__main__':
    unittest.main()

