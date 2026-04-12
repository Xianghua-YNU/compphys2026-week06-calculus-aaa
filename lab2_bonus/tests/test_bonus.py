import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from bonus_plate_gravity import force_curve, gauss_legendre_2d, plate_force_z


class TestBonus(unittest.TestCase):
    def test_gauss_2d_constant_points_10(self):
        try:
            val = gauss_legendre_2d(lambda x, y: 1.0, -1.0, 1.0, -2.0, 2.0, n=20)
        except NotImplementedError as exc:
            self.fail(f"TODO D1 未完成: {exc}")
        self.assertLess(abs(val - 8.0), 1e-10)

    def test_force_positive_points_10(self):
        try:
            fz = plate_force_z(1.0, L=10.0, M_plate=1.0e4, m_particle=1.0, n=30)
        except NotImplementedError as exc:
            self.fail(f"TODO D2 未完成: {exc}")
        self.assertGreater(fz, 0.0)

    def test_force_curve_shape_points_10(self):
        z_vals = np.array([0.2, 1.0, 5.0])
        try:
            f_vals = force_curve(z_vals, L=10.0, M_plate=1.0e4, m_particle=1.0, n=20)
        except NotImplementedError as exc:
            self.fail(f"TODO D3 未完成: {exc}")
        self.assertEqual(f_vals.shape, z_vals.shape)


if __name__ == "__main__":
    unittest.main()
