import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from task_c_ring_potential import axis_potential_analytic, ring_potential_grid, ring_potential_point


class TestTaskC(unittest.TestCase):
    def test_point_potential_finite_points_8(self):
        try:
            v = ring_potential_point(0.0, 0.0, 0.5, a=1.0, q=1.0, n_phi=720)
        except NotImplementedError as exc:
            self.fail(f"TODO C1 未完成: {exc}")
        self.assertTrue(np.isfinite(v))
        self.assertGreater(v, 0.0)

    def test_axis_consistency_points_8(self):
        z = 0.8
        v_true = axis_potential_analytic(z, a=1.0, q=1.0)
        try:
            v_num = ring_potential_point(0.0, 0.0, z, a=1.0, q=1.0, n_phi=2000)
        except NotImplementedError as exc:
            self.fail(f"TODO C1 未完成: {exc}")
        self.assertLess(abs(v_num - v_true), 5e-3)

    def test_grid_shape_points_7(self):
        ys = np.linspace(-0.5, 0.5, 11)
        zs = np.linspace(-0.5, 0.5, 13)
        try:
            V = ring_potential_grid(ys, zs, x0=0.0, a=1.0, q=1.0, n_phi=360)
        except NotImplementedError as exc:
            self.fail(f"TODO C2 未完成: {exc}")
        self.assertEqual(V.shape, (len(zs), len(ys)))


if __name__ == "__main__":
    unittest.main()
