import unittest
import fairlib as fl
import pandas.testing as pd_testing

from fairlib.preprocessing import Reweighing, ReweighingWithMean


class TestReweighing(unittest.TestCase):
    """Test class for the Reweighing and ReweighingWithMean fairness-aware preprocessing algorithms."""

    TARGET1 = "target1"
    TARGET2 = "target2"
    SENSITIVE1 = "sensitive1"
    SENSITIVE2 = "sensitive2"

    def setUp(self):
        """Set up the test case with a simple dataset containing target and sensitive attributes."""
        self.df = fl.DataFrame(
            {
                self.TARGET1: [0, 1, 0, 1, 0, 1],
                self.TARGET2: [1, 0, 1, 0, 1, 0],
                self.SENSITIVE1: [0, 1, 0, 1, 0, 0],
                self.SENSITIVE2: [1, 1, 0, 0, 0, 1],
            }
        )

    def test_single_reweighing(self):
        """Test Reweighing with a single target and sensitive attribute."""
        self.df.targets = {self.TARGET1}
        self.df.sensitive = {self.SENSITIVE1}
        model = Reweighing()
        transformed_df = model.fit_transform(self.df)
        expected_df = fl.DataFrame(
            {
                self.TARGET1: [0, 1, 0, 1, 0, 1],
                self.TARGET2: [1, 0, 1, 0, 1, 0],
                self.SENSITIVE1: [0, 1, 0, 1, 0, 0],
                self.SENSITIVE2: [1, 1, 0, 0, 0, 1],
                "weights": [0.666667, 0.500000, 0.666667, 0.500000, 0.666667, 2.000000],
            }
        )
        pd_testing.assert_frame_equal(transformed_df, expected_df)

        # Test with multiple targets (should raise ValueError)
        self.df.targets = {self.TARGET1, self.TARGET2}
        self.df.sensitive = {self.SENSITIVE1}

        with self.assertRaises(ValueError):
            model.fit_transform(self.df)

    def test_multiple_sensitive_reweighing(self):
        """Test Reweighing with multiple sensitive attributes."""
        # Test for multiple sensitive columns in Reweighing
        self.df.targets = {self.TARGET1}
        self.df.sensitive = {self.SENSITIVE1, self.SENSITIVE2}
        model = Reweighing()
        transformed_df = model.fit_transform(self.df)

        # Expected DataFrame with combined effect of multiple sensitive columns
        expected_df = fl.DataFrame(
            {
                self.TARGET1: [0, 1, 0, 1, 0, 1],
                self.TARGET2: [1, 0, 1, 0, 1, 0],
                self.SENSITIVE1: [0, 1, 0, 1, 0, 0],
                self.SENSITIVE2: [1, 1, 0, 0, 0, 1],
                "weights": [1.0, 0.5, 0.5, 1.0, 0.5, 1.0],
            }
        )

        # Assert DataFrame equality
        pd_testing.assert_frame_equal(transformed_df, expected_df)

    def test_reweighing_with_mean(self):
        """Test ReweighingWithMean with multiple sensitive attributes."""
        # Test for ReweighingWithMean with multiple sensitive columns
        self.df.targets = {self.TARGET1}
        self.df.sensitive = {self.SENSITIVE1, self.SENSITIVE2}
        model = ReweighingWithMean()
        transformed_df = model.fit_transform(self.df)

        # Expected DataFrame for ReweighingWithMean method
        expected_df = fl.DataFrame(
            {
                self.TARGET1: [0, 1, 0, 1, 0, 1],
                self.TARGET2: [1, 0, 1, 0, 1, 0],
                self.SENSITIVE1: [0, 1, 0, 1, 0, 0],
                self.SENSITIVE2: [1, 1, 0, 0, 0, 1],
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

    def test_reweighing_with_mean_single_sensitive(self):
        """Test ReweighingWithMean with a single sensitive attribute."""
        # Test with single sensitive column
        self.df.targets = {self.TARGET1}
        self.df.sensitive = {self.SENSITIVE1}
        model = ReweighingWithMean()
        transformed_df_single_sensitive = model.fit_transform(self.df)

        expected_df_single_sensitive = fl.DataFrame(
            {
                self.TARGET1: [0, 1, 0, 1, 0, 1],
                self.TARGET2: [1, 0, 1, 0, 1, 0],
                self.SENSITIVE1: [0, 1, 0, 1, 0, 0],
                self.SENSITIVE2: [1, 1, 0, 0, 0, 1],
                "weights": [0.666667, 0.500000, 0.666667, 0.500000, 0.666667, 2.000000],
            }
        )

        # Assert DataFrame equality for single sensitive column
        pd_testing.assert_frame_equal(
            transformed_df_single_sensitive, expected_df_single_sensitive
        )

    def test_reweighing_with_mean_error_handling(self):
        """Test error handling for ReweighingWithMean with multiple target columns."""
        # Test for multiple target columns (should raise ValueError)
        self.df.targets = {self.TARGET1, self.TARGET2}
        self.df.sensitive = {self.SENSITIVE1, self.SENSITIVE2}

        with self.assertRaises(ValueError):
            model = ReweighingWithMean()
            transformed_df = model.fit_transform(self.df)

    def tearDown(self):
        del self.df


if __name__ == "__main__":
    unittest.main()
