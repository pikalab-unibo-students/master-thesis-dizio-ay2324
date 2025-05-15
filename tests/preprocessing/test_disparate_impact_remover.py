import unittest
import numpy as np
import torch
import fairlib as fl
import pandas.testing as pd_testing
from sklearn.model_selection import train_test_split

from fairlib.preprocessing import DisparateImpactRemover
from tests.data_generator import biased_dataset_people_height
from tests.preprocessing.test_utils import set_seed, train_classifier, evaluate_fairness


class TestDisparateImpactRemover(unittest.TestCase):
    """Test class for the DisparateImpactRemover fairness-aware preprocessing algorithm."""

    TARGET = "class"
    SENSITIVE = "male"

    def setUp(self):
        """Set up the test case by loading the dataset and setting target/sensitive attributes."""
        # Use the biased_dataset_people_height dataset
        set_seed(42)
        dataset = biased_dataset_people_height(binary=True)
        self.df = fl.DataFrame(dataset)

        # Set target and sensitive attributes
        self.df.targets = self.TARGET
        self.df.sensitive = self.SENSITIVE

    def test_fit_transform(self):
        """Test fit_transform with default repair level (1.0)"""
        dir_model = DisparateImpactRemover()
        transformed_df = dir_model.fit_transform(self.df)

        # Verify the transformed DataFrame has the expected columns
        self.assertTrue("height" in transformed_df.columns)
        self.assertTrue("weight" in transformed_df.columns)

        # Verify sensitive attribute metadata is preserved
        self.assertEqual(transformed_df.sensitive, self.df.sensitive)

        # Verify the transformation has occurred (values should be different)
        self.assertFalse(
            np.array_equal(transformed_df["height"].values, self.df["height"].values)
        )
        self.assertFalse(
            np.array_equal(transformed_df["weight"].values, self.df["weight"].values)
        )

    def test_no_repair(self):
        """Test with repair_level=0.0 (no repair)"""
        dir_model = DisparateImpactRemover(repair_level=0.0)
        transformed_df = dir_model.fit_transform(self.df)

        # With repair_level=0.0, the feature values should be identical to the original
        # Note: The transformed DataFrame might have different metadata, so we only compare feature values
        self.assertTrue(
            np.array_equal(transformed_df["height"].values, self.df["height"].values)
        )
        self.assertTrue(
            np.array_equal(transformed_df["weight"].values, self.df["weight"].values)
        )

    def test_partial_repair(self):
        """Test with repair_level=0.5 (partial repair)"""
        dir_model = DisparateImpactRemover(repair_level=0.5)
        transformed_df = dir_model.fit_transform(self.df)

        # With partial repair, values should be between original and fully repaired
        full_repair_df = DisparateImpactRemover(repair_level=1.0).fit_transform(self.df)

        # Check height values are between original and fully repaired
        for i in range(len(self.df)):
            original_val = self.df["height"].values[i]
            transformed_val = transformed_df["height"].values[i]
            full_repair_val = full_repair_df["height"].values[i]

            # If original and full repair values are different
            if original_val != full_repair_val:
                # Check if transformed value is between original and full repair
                # or equal to one of them (due to floating point precision)
                self.assertTrue(
                    (original_val <= transformed_val <= full_repair_val)
                    or (full_repair_val <= transformed_val <= original_val)
                    or np.isclose(transformed_val, original_val)
                    or np.isclose(transformed_val, full_repair_val)
                )

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        dir_model = DisparateImpactRemover()
        with self.assertRaises(TypeError):
            dir_model.fit_transform(np.array([1, 2, 3]))

        # Test with multiple sensitive attributes
        multi_sensitive_df = self.df.copy()
        multi_sensitive_df.sensitive = {
            "male",
            "class",
        }  # Using class as a second sensitive attribute
        with self.assertRaises(ValueError):
            dir_model.fit_transform(multi_sensitive_df)

    def test_transformation_effect(self):
        """Test the effect of transformation on feature values"""
        # Split the dataset to create a test subset
        X_train, X_test = train_test_split(self.df, test_size=0.3, random_state=42)

        # Apply the transformation with full repair
        dir_model = DisparateImpactRemover(repair_level=1.0)
        transformed_df = dir_model.fit_transform(X_train)

        # Verify that the transformation has occurred by checking that values have changed
        for feature in ["height", "weight", "income"]:
            # Get original feature values
            orig_values = X_train[feature].values
            # Get transformed feature values
            trans_values = transformed_df[feature].values

            # Verify that at least some values have changed
            self.assertFalse(
                np.array_equal(orig_values, trans_values),
                f"Expected {feature} values to change after transformation",
            )

        # Verify that the model attributes are properly set
        self.assertIsNotNone(dir_model.sensitive_values)
        self.assertIsNotNone(dir_model.quantile_maps)
        self.assertTrue(len(dir_model.quantile_maps) > 0)

        # Test fit_transform on new data (DisparateImpactRemover doesn't have a separate transform method)
        transformed_test = dir_model.fit_transform(X_test)
        self.assertEqual(len(transformed_test), len(X_test))

        # Check that important feature columns are present (not necessarily all columns)
        feature_columns = ["height", "weight", "income"]
        self.assertTrue(all(col in transformed_test.columns for col in feature_columns))

    def test_fairness_improvement(self):
        """Test if DisparateImpactRemover improves fairness metrics"""
        # Split data
        X = self.df.copy()
        y = X.pop(self.TARGET)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train classifier on original data
        clf_original = train_classifier(X_train, y_train)
        y_pred_original = clf_original.predict(X_test)

        # Apply DisparateImpactRemover
        dir_model = DisparateImpactRemover(repair_level=1.0)
        X_train_transformed = dir_model.fit_transform(X_train)
        X_test_transformed = dir_model.fit_transform(X_test)

        # Train classifier on transformed data
        clf_transformed = train_classifier(X_train_transformed, y_train)
        y_pred_transformed = clf_transformed.predict(X_test_transformed)

        # Evaluate fairness metrics
        spd_original, di_original = evaluate_fairness(X_test.copy(), y_pred_original, self.TARGET, self.SENSITIVE)
        spd_transformed, di_transformed = evaluate_fairness(
            X_test.copy(), y_pred_transformed, self.TARGET, self.SENSITIVE
        )

        # Check if fairness improved
        for key in spd_original:
            self.assertLessEqual(
                abs(spd_transformed[key]),
                abs(spd_original[key]),
                f"SPD for {key} should be improved with DisparateImpactRemover",
            )
        for key in di_original:
            self.assertLessEqual(
                abs(di_transformed[key] - 1),
                abs(di_original[key] - 1),
                f"DI for {key} should be improved with DisparateImpactRemover",
            )

    def tearDown(self):
        del self.df


if __name__ == "__main__":
    unittest.main()
