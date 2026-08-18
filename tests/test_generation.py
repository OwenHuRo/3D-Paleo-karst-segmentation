import unittest

import numpy as np

from ConstructData.CreateProcess import generate_synthetic_seismic_final


class SyntheticGenerationSmokeTests(unittest.TestCase):
    def test_generation_is_reproducible(self):
        parameters = dict(
            nx=16,
            ny=16,
            nt=16,
            num_layers=4,
            num_faults=1,
            num_anomalies=2,
            dip_angle=20,
            save_intermediate_dat=False,
            seed=12345
        )

        seismic_1, label_1, anomalies_1, _ = generate_synthetic_seismic_final(**parameters)
        seismic_2, label_2, anomalies_2, _ = generate_synthetic_seismic_final(**parameters)

        self.assertEqual(seismic_1.shape, (16, 16, 16))
        self.assertEqual(label_1.shape, (16, 16, 16))
        np.testing.assert_array_equal(seismic_1, seismic_2)
        np.testing.assert_array_equal(label_1, label_2)
        self.assertEqual(anomalies_1, anomalies_2)

    def test_invalid_dip_angle_is_rejected(self):
        with self.assertRaises(ValueError):
            generate_synthetic_seismic_final(
                nx=16,
                ny=16,
                nt=16,
                num_layers=4,
                num_faults=1,
                num_anomalies=1,
                dip_angle=0,
                save_intermediate_dat=False
            )


if __name__ == '__main__':
    unittest.main()

