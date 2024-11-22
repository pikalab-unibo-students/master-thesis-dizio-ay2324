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
        col3 = str_to_int_list("11010101")
        self.data = array([col1, col2, col3]).transpose()

    def test_statistical_parity_difference(self):
        spd = statistical_parity_difference(
            target_column=self.data[:, 1],
            sensitive_column=self.data[:, 0]
        )
        self.assertTrue(np.array_equal(spd.squeeze(), np.array([-0.5, 0.5])))

    def test_equality_of_opportunity(self):
        eoo = equality_of_opportunity(
            target_column=self.data[:, 1],
            sensitive_column=self.data[:, 0],
            predicted_column=self.data[:, 2]
        )

        self.assertTrue(np.allclose(eoo.squeeze(), np.array([3.66666667, -3.66666667])))


class TestMetricsOnVeryPolarisedBinaryData(unittest.TestCase):

    def setUp(self) -> None:
        col1 = str_to_int_list("00000000")
        col2 = str_to_int_list("11111111")
        col3 = str_to_int_list("01011010")
        col4 = str_to_int_list("11010101")
        self.data = array([col1, col2, col3, col4]).transpose()

    def test_statistical_parity_difference_with_all_zero(self):
        spd = statistical_parity_difference(
            target_column=self.data[:, 2],
            sensitive_column=self.data[:, 0]
        )
        self.assertEqual(spd, inf)

    def test_statistical_parity_difference_with_all_one(self):
        spd = statistical_parity_difference(
            target_column=self.data[:, 2],
            sensitive_column=self.data[:, 1]
        )
        self.assertEqual(spd, inf)

    def test_equality_of_opportunity_with_all_zero(self):
        eoo = equality_of_opportunity(
            target_column=self.data[:, 2],
            sensitive_column=self.data[:, 0],
            predicted_column=self.data[:, 3]
        )
        self.assertEqual(eoo, inf)

    def test_equality_of_opportunity_with_all_one(self):
        eoo = equality_of_opportunity(
            target_column=self.data[:, 2],
            sensitive_column=self.data[:, 1],
            predicted_column=self.data[:, 3]
        )
        self.assertEqual(eoo, inf)