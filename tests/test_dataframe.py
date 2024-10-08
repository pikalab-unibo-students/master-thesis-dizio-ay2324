import unittest
import fairlib as fl
import pandas as pd


class TestCustomDataframe(unittest.TestCase):

    def setUp(self):
        self.df = fl.DataFrame(
            {
                "name": ["Alice", "Bob", "Carla", "Davide", "Elena"],
                "age": [25, 32, 45, 29, 34],
                "sex": ["F", "M", "F", "M", "F"],
                "income": ["40000", "50000", "45000", "53000", "43000"],
            }
        )
        self.df.targets = "income"
        self.df.sensitive = {"age", "sex"}

    def testName(self):
        expected_names = pd.Series(["Alice", "Bob", "Carla", "Davide", "Elena"])
        pd.testing.assert_series_equal(
            expected_names, self.df["name"], check_names=False
        )

    def testTargets(self):
        self.assertEqual({"income"}, self.df.targets)

        with self.assertRaises(Exception):
            self.df.targets = {"missing"}

    def testSensitive(self):
        self.assertEqual({"age", "sex"}, self.df.sensitive)

        with self.assertRaises(Exception):
            self.df.sensitive = {"age", "sex", "missing"}

    def testDrop(self):
        df2 = self.df.drop(["name"], axis=1)
        self.assertEqual({"income"}, df2.targets)

        df3 = self.df.drop(["sex"], axis=1)
        self.assertEqual({"age"}, df3.sensitive)

    def tearDown(self):
        del self.df


if __name__ == "__main__":
    unittest.main()
