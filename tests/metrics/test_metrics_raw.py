import numpy as np
from numpy import array, inf
from fairlib.metrics import *
import unittest


def str_to_int_list(x: str) -> list:
    return [int(i) for i in x]


class TestMetricsOnBinaryData(unittest.TestCase):
    def setUp(self) -> None:
        col1 = str_to_int_list("10010101")
        col2 = str_to_int_list("01011010")
        self.data = array([col1, col2]).transpose()

    def test_statistical_parity_difference(self):
        spd = statistical_parity_difference(
            target_column=self.data[:, 1],
            sensitive_column=self.data[:, 0]
        )
        print(spd)
        self.assertTrue(np.array_equal(spd.squeeze(), np.array([-0.5, 0.5])))


class TestMetricsOnVeryPolarisedBinaryData(unittest.TestCase):

    def test_statistical_parity_difference_with_all_zero(self):
        col1 = str_to_int_list("00000000")
        col2 = str_to_int_list("01011010")
        self.data = array([col1, col2]).transpose()
        spd = statistical_parity_difference(
            target_column=self.data[:, 1],
            sensitive_column=self.data[:, 0]
        )
        self.assertEqual(spd, inf)

    def test_statistical_parity_difference_with_all_one(self):
        col1 = str_to_int_list("11111111")
        col2 = str_to_int_list("01011010")
        self.data = array([col1, col2]).transpose()
        spd = statistical_parity_difference(
            target_column=self.data[:, 1],
            sensitive_column=self.data[:, 0]
        )
        self.assertEqual(spd, inf)
