import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from task_b_integration import debye_integral, simpson_composite, trapezoid_composite


class TestTaskB(unittest.TestCase):
    def test_trapezoid_polynomial_points_8(self):
        f = lambda x: x**2
        try:
            val = trapezoid_composite(f, 0.0, 1.0, 2000)
        except NotImplementedError as exc:
            self.fail(f"TODO B1 未完成: {exc}")
        self.assertLess(abs(val - 1.0 / 3.0), 1e-4)

    def test_simpson_even_constraint_points_8(self):
        try:
            simpson_composite(lambda x: x, 0.0, 1.0, 101)
        except NotImplementedError as exc:
            self.fail(f"TODO B2 未完成: {exc}")
        except ValueError:
            return
        self.fail("simpson_composite 应在 n 为奇数时抛出 ValueError")

    def test_simpson_outperforms_trap_points_8(self):
        f = lambda x: x**4
        try:
            t = trapezoid_composite(f, 0.0, 1.0, 100)
        except NotImplementedError as exc:
            self.fail(f"TODO B1 未完成: {exc}")
        try:
            s = simpson_composite(f, 0.0, 1.0, 100)
        except NotImplementedError as exc:
            self.fail(f"TODO B2 未完成: {exc}")
        self.assertLess(abs(s - 0.2), abs(t - 0.2))

    def test_debye_integral_positive_points_8(self):
        try:
            v = debye_integral(80.0, theta_d=428.0, method="simpson", n=200)
        except NotImplementedError as exc:
            self.fail(f"TODO B3 未完成: {exc}")
        self.assertGreater(v, 0.0)


if __name__ == "__main__":
    unittest.main()
