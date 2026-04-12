import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from task_a_nuclear_sensitivity import rate_3alpha, sensitivity_nu, nu_table


class TestTaskA(unittest.TestCase):
    def test_rate_positive_points_7(self):
        self.assertGreater(rate_3alpha(1.0e8), 0.0)

    def test_nu_reasonable_range_points_8(self):
        try:
            nu = sensitivity_nu(1.0e8, h=1e-8)
        except NotImplementedError as exc:
            self.fail(f"TODO A2 未完成: {exc}")
        self.assertTrue(math.isfinite(nu))
        self.assertGreater(nu, 1.0)

    def test_nu_table_shape_points_8(self):
        T_values = [1.0e8, 2.5e8, 5.0e8]
        try:
            tbl = nu_table(T_values, h=1e-8)
        except NotImplementedError as exc:
            self.fail(f"TODO A3 未完成: {exc}")
        self.assertEqual(len(tbl), len(T_values))
        self.assertEqual(len(tbl[0]), 2)


if __name__ == "__main__":
    unittest.main()
