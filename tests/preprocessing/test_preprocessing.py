import unittest
import fairlib as fl
import fairlib.preprocessing.pre_processing
import pandas.testing as pd_testing


class TestPreProcessing(unittest.TestCase):

    def setUp(self):
        self.df = fl.DataFrame(
            {
                "target1": [0, 1, 0, 1, 0, 1],
                "target2": [1, 0, 1, 0, 1, 0],
                "sensitive1": [0, 1, 0, 1, 0, 0],
                "sensitive2": [1, 1, 0, 0, 0, 1],
            }
        )

    def testSingleReweighing(self):
        self.df.targets = {"target1"}
        self.df.sensitive = {"sensitive1"}
        transformed_df = self.df.reweighing()
        expected_df = fl.DataFrame(
            {
                "target1": [0, 1, 0, 1, 0, 1],
                "target2": [1, 0, 1, 0, 1, 0],
                "sensitive1": [0, 1, 0, 1, 0, 0],
                "sensitive2": [1, 1, 0, 0, 0, 1],
                "weights": [0.666667, 0.500000, 0.666667, 0.500000, 0.666667, 2.000000],
            }
        )
        pd_testing.assert_frame_equal(transformed_df, expected_df)

        self.df.targets = {"target1", "target2"}
        self.df.sensitive = {"sensitive1"}

        with self.assertRaises(ValueError):
            self.df.reweighing()

    def testMultipleSensitiveReweighing(self):
        # Test for multiple sensitive columns in Reweighing
        self.df.targets = {"target1"}
        self.df.sensitive = {"sensitive1", "sensitive2"}
        transformed_df = self.df.reweighing()

        # Expected DataFrame with combined effect of multiple sensitive columns
        expected_df = fl.DataFrame(
            {
                "target1": [0, 1, 0, 1, 0, 1],
                "target2": [1, 0, 1, 0, 1, 0],
                "sensitive1": [0, 1, 0, 1, 0, 0],
                "sensitive2": [1, 1, 0, 0, 0, 1],
                "weights": [1.0, 0.5, 0.5, 1.0, 0.5, 1.0],
            }
        )

        # Assert DataFrame equality
        pd_testing.assert_frame_equal(transformed_df, expected_df)

    def testReweighingWithMean(self):
        # Test for ReweighingWithMean with multiple sensitive columns
        self.df.targets = {"target1"}
        self.df.sensitive = {"sensitive1", "sensitive2"}
        transformed_df = self.df.reweighing_with_mean()

        # Expected DataFrame for ReweighingWithMean method
        expected_df = fl.DataFrame(
            {
                "target1": [0, 1, 0, 1, 0, 1],
                "target2": [1, 0, 1, 0, 1, 0],
                "sensitive1": [0, 1, 0, 1, 0, 0],
                "sensitive2": [1, 1, 0, 0, 0, 1],
                "weights": [
                    1.0833333333333333,
                    0.625,
                    0.7083333333333333,
                    1.0,
                    0.7083333333333333,
                    1.375,
                ],
            }
        )

        # Assert DataFrame equality
        pd_testing.assert_frame_equal(transformed_df, expected_df)

        # Test with single sensitive column
        self.df.sensitive = {"sensitive1"}
        transformed_df_single_sensitive = self.df.reweighing_with_mean()

        expected_df_single_sensitive = fl.DataFrame(
            {
                "target1": [0, 1, 0, 1, 0, 1],
                "target2": [1, 0, 1, 0, 1, 0],
                "sensitive1": [0, 1, 0, 1, 0, 0],
                "sensitive2": [1, 1, 0, 0, 0, 1],
                "weights": [0.666667, 0.500000, 0.666667, 0.500000, 0.666667, 2.000000],
            }
        )

        # Assert DataFrame equality for single sensitive column
        pd_testing.assert_frame_equal(
            transformed_df_single_sensitive, expected_df_single_sensitive
        )

        # Test for multiple target columns (should raise ValueError)
        self.df.targets = {"target1", "target2"}
        self.df.sensitive = {"sensitive1", "sensitive2"}

        with self.assertRaises(ValueError):
            self.df.reweighing_with_mean()

    def tearDown(self):
        del self.df


if __name__ == "__main__":
    unittest.main()
