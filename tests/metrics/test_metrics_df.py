import unittest
import fairlib as fl
from fairlib.utils import DomainDict, Assignment


class TestMetrics(unittest.TestCase):

    def setUp(self):
        self.df = fl.DataFrame(
            {
                "target1": [1, 0, 1, 1, 0, 1, 0, 0],
                "target2": [0, 1, 0, 1, 1, 0, 1, 1],
                "sensitive1": [1, 1, 0, 0, 1, 0, 0, 1],
                "sensitive2": [0, 0, 1, 1, 0, 1, 1, 0],
                "sensitive3": [1, 0, 1, 1, 1, 0, 0, 1],
                "predictions": [0, 0, 1, 0, 0, 1, 0, 1],
            }
        )

    def testStatisticalParityDifference(self):
        self.df.targets = ["target1"]
        self.df.sensitive = ["sensitive1"]
        res = {(Assignment("target1", 1), Assignment("sensitive1", 1)): -0.5,
               (Assignment("target1", 1), Assignment("sensitive1", 0)): 0.5}
        expected_spd = DomainDict(res)

        spd_result = self.df.statistical_parity_difference()
        assert (
            spd_result == expected_spd
        ), f"Expected {expected_spd}, but got {spd_result}"

        self.df.targets = ["target1", "target2"]
        self.df.sensitive = ["sensitive1", "sensitive2", "sensitive3"]

        res = {
            (Assignment("target1", 1), Assignment("sensitive1", 1)): -0.5,
            (Assignment("target1", 1), Assignment("sensitive1", 0)): 0.5,
            (Assignment("target1", 1), Assignment("sensitive2", 1)): 0.5,
            (Assignment("target1", 1), Assignment("sensitive2", 0)): -0.5,
            (Assignment("target1", 1), Assignment("sensitive3", 1)): 0.26666666666666666,
            (Assignment("target1", 1), Assignment("sensitive3", 0)): -0.26666666666666666,
            (Assignment("target2", 1), Assignment("sensitive1", 1)): 0.25,
            (Assignment("target2", 1), Assignment("sensitive1", 0)): -0.25,
            (Assignment("target2", 1), Assignment("sensitive2", 1)): -0.25,
            (Assignment("target2", 1), Assignment("sensitive2", 0)): 0.25,
            (Assignment("target2", 1), Assignment("sensitive3", 1)): -0.06666666666666665,
            (Assignment("target2", 1), Assignment("sensitive3", 0)): 0.06666666666666665,
        }
        expected_spd = DomainDict(res)

        spd_result = self.df.statistical_parity_difference()
        assert (
                spd_result == expected_spd
        ), f"Expected {expected_spd}, but got {spd_result}"

    def testDisparateImpact(self):
        self.df.targets = ["target1"]
        self.df.sensitive = ["sensitive1"]
        res = {(Assignment("target1", 1), Assignment("sensitive1", 1)): 3.0,
               (Assignment("target1", 1), Assignment("sensitive1", 0)): 0.3333333333333333}
        expected_di = DomainDict(res)

        di_result = self.df.disparate_impact()
        assert di_result == expected_di, f"Expected {expected_di}, but got {di_result}"

        self.df.targets = ["target1", "target2"]
        self.df.sensitive = ["sensitive1", "sensitive2", "sensitive3"]

        res = {
            (Assignment("target1", 1), Assignment("sensitive1", 1)): 3.0,
            (Assignment("target1", 1), Assignment("sensitive1", 0)): 0.3333333333333333,
            (Assignment("target1", 1), Assignment("sensitive2", 1)): 0.3333333333333333,
            (Assignment("target1", 1), Assignment("sensitive2", 0)): 3.0,
            (Assignment("target1", 1), Assignment("sensitive3", 1)): 0.5555555555555556,
            (Assignment("target1", 1), Assignment("sensitive3", 0)): 1.8,
            (Assignment("target2", 1), Assignment("sensitive1", 1)): 0.6666666666666666,
            (Assignment("target2", 1), Assignment("sensitive1", 0)): 1.5,
            (Assignment("target2", 1), Assignment("sensitive2", 1)): 1.5,
            (Assignment("target2", 1), Assignment("sensitive2", 0)): 0.6666666666666666,
            (Assignment("target2", 1), Assignment("sensitive3", 1)): 1.1111111111111112,
            (Assignment("target2", 1), Assignment("sensitive3", 0)): 0.9,
        }

        expected_di = DomainDict(res)

        di_result = self.df.disparate_impact()
        assert (
                di_result == expected_di
        ), f"Expected {expected_di}, but got {di_result}"

    def testEqualityOfOpportunity(self):
        self.df.targets = ["target1"]
        self.df.sensitive = ["sensitive1"]
        predictions = self.df["predictions"]
        res = {(Assignment("target1", 1), Assignment("sensitive1", 1)): 0.33333333333333337,
               (Assignment("target1", 1), Assignment("sensitive1", 0)): -0.33333333333333337}
        expected_eoo = DomainDict(res)

        eoo_result = self.df.equality_of_opportunity(predictions)
        assert eoo_result == expected_eoo, f"Expected {expected_eoo}, but got {eoo_result}"

        self.df.targets = ["target1", "target2"]
        self.df.sensitive = ["sensitive1", "sensitive2", "sensitive3"]

        res = {
            (Assignment("target1", 1), Assignment("sensitive3", 1)): -0.33333333333333337,
            (Assignment("target1", 1), Assignment("sensitive3", 0)): 0.33333333333333337,
            (Assignment("target1", 1), Assignment("sensitive1", 1)): 0.33333333333333337,
            (Assignment("target1", 1), Assignment("sensitive1", 0)): -0.33333333333333337,
            (Assignment("target1", 1), Assignment("sensitive2", 1)): -0.33333333333333337,
            (Assignment("target1", 1), Assignment("sensitive2", 0)): 0.33333333333333337,
            (Assignment("target2", 1), Assignment("sensitive3", 1)): 0.16666666666666663,
            (Assignment("target2", 1), Assignment("sensitive3", 0)): -0.16666666666666663,
            (Assignment("target2", 1), Assignment("sensitive1", 1)): -0.6666666666666667,
            (Assignment("target2", 1), Assignment("sensitive1", 0)): 0.6666666666666667,
            (Assignment("target2", 1), Assignment("sensitive2", 1)): 0.6666666666666667,
            (Assignment("target2", 1), Assignment("sensitive2", 0)): -0.6666666666666667,
        }
        expected_eoo = DomainDict(res)

        eoo_result = self.df.equality_of_opportunity(predictions)
        assert (
                eoo_result == expected_eoo
        ), f"Expected {expected_eoo}, but got {eoo_result}"

    def tearDown(self):
        del self.df


if __name__ == "__main__":
    unittest.main()
