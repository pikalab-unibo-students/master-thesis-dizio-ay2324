import unittest
import fairlib as fl
import pandas as pd


class TestCustomDataframe(unittest.TestCase):
    def assertDataFramesAreEqual(self, df1, df2):
        self.assertTrue(df1.equals(df2))

    def setUp(self):
        self.df = fl.DataFrame(
            {
                "name": ["Alice", "Bob", "Carla", "Davide", "Elena"],
                "age": [25, 32, 45, 29, 34],
                "sex": ["F", "M", "F", "M", "F"],
                "income": ["40000", "50000", "45000", "53000", "43000"],
            }
        ).reindex(columns=["name", "age", "sex", "income"], copy=False)
        self.df.targets = "income"
        self.df.sensitive = {"age", "sex"}
        self.discretized = fl.DataFrame(
            {
                "name": ["Alice", "Bob", "Carla", "Davide", "Elena"],
                "age>=30": [False, True, True, False, True],
                "sex==M": [False, True, False, True, False],
                "sex==F": [True, False, True, False, True],
                "income>=50k": [False, True, False, True, False],
            }
        ).reindex(columns=["name", "age>=30", "sex==F", "sex==M", "income>=50k"])

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

    def testIsDiscrete(self):
        self.assertFalse(self.df.is_binary())
        self.assertTrue(self.df.is_discrete())
        self.assertTrue(self.discretized.drop(["name"], axis=1).is_binary())

    def testDiscretize(self):
        discretized = self.df.discretize(
            ("age>=30", self.df["age"] >= 30),
            sex="ohe",
            income=("income>=50k", lambda income: int(income) >= 50_000),
        )
        self.assertDataFramesAreEqual(self.discretized, discretized)

    def testDiscretizePreservesTargetsAndSensitive(self):
        discretized = self.df.discretize(
            ("age>=30", self.df["age"] >= 30),
            sex="ohe",
            income=("income>=50k", lambda income: int(income) >= 50_000),
        )
        self.assertEqual({"income>=50k"}, discretized.targets)
        self.assertEqual({"age>=30", "sex==F", "sex==M"}, discretized.sensitive)

    def testSeparateColumns(self):
        X, y = self.df.separate_columns()
        self.assertDataFramesAreEqual(X, self.df.drop(["income"], axis=1))
        self.assertDataFramesAreEqual(y, fl.DataFrame({"income": self.df["income"]}))

    def tearDown(self):
        del self.df


if __name__ == "__main__":
    unittest.main()
