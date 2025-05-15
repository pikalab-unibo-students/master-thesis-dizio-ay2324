import unittest
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from fairlib.preprocessing.lfr import LFR
from fairlib import DataFrame
from tests.data_generator import biased_dataset_people_height
from tests.preprocessing.test_utils import set_seed, train_classifier, evaluate_fairness


class TestLFR(unittest.TestCase):
    """
    Test class for the LFR (Learning Fair Representations) fairness-aware learning model.
    """

    EPOCHS = 20
    TARGET = "class"
    SENSITIVE = "male"

    def setUp(self):
        """
        Set up the test case by loading the dataset and initializing the input features and target labels.
        """
        set_seed(42)
        dataset = biased_dataset_people_height(binary=True)
        self.df = DataFrame(dataset)

        # Set target and sensitive attributes
        self.df.targets = self.TARGET
        self.df.sensitive = self.SENSITIVE

    def test_fit_transform(self):
        """Test the fit_transform method with default parameters"""
        latent_dim = 8
        lfr_model = LFR(
            input_dim=self.df.shape[1] - 1,  # Exclude target column
            latent_dim=latent_dim,
            output_dim=self.df.shape[1] - 1,
            alpha_z=1.0,
            alpha_x=1.0,
            alpha_y=1.0,
        )
        transformed_df = lfr_model.fit_transform(self.df, epochs=self.EPOCHS)

        # Verify the transformed DataFrame has the expected shape
        self.assertEqual(transformed_df.shape[0], self.df.shape[0])
        self.assertEqual(transformed_df.shape[1], latent_dim)  # latent_dim
        
        # Verify column names follow the expected pattern
        expected_cols = [f'z{i}' for i in range(latent_dim)]
        self.assertListEqual(list(transformed_df.columns), expected_cols)

    def test_error_handling(self):
        """Test error handling for invalid inputs"""
        lfr_model = LFR(
            input_dim=5,
            latent_dim=8,
            output_dim=5
        )
        with self.assertRaises(TypeError):
            lfr_model.fit_transform(np.array([1, 2, 3]))

        # Test with multiple sensitive attributes
        multi_sensitive_df = self.df.copy()
        multi_sensitive_df.sensitive = {
            "male",
            "class",
        }  # Using class as a second sensitive attribute
        with self.assertRaises(ValueError):
            lfr_model.fit_transform(multi_sensitive_df)

    def test_fairness_improvement(self):
        """Test if LFR improves fairness metrics"""
        # Split data
        X = self.df.copy()
        y = X.pop(self.TARGET)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Prepare data for original classifier
        X_train_orig = X_train.copy()
        X_test_orig = X_test.copy()

        # Train classifier on original data
        clf_original = train_classifier(X_train_orig, y_train)
        y_pred_original = clf_original.predict(X_test_orig)

        # Prepare data for LFR
        X_train_with_target = X_train.copy()
        X_train_with_target[self.TARGET] = y_train
        train_dataset = DataFrame(X_train_with_target)
        train_dataset.targets = self.TARGET
        train_dataset.sensitive = self.SENSITIVE

        # Initialize and train LFR model
        lfr = LFR(
            input_dim=X_train.shape[1],
            latent_dim=8,
            output_dim=X_train.shape[1],
            alpha_z=1.0,
            alpha_x=1.0,
            alpha_y=1.0,
        )

        # Transform the training data using LFR
        X_train_transformed = lfr.fit_transform(train_dataset, epochs=self.EPOCHS)

        # Transform the test data using LFR (use transform instead of fit_transform)
        X_test_transformed = lfr.transform(X_test)

        # Train classifier on transformed data
        clf_transformed = train_classifier(X_train_transformed, y_train)
        y_pred_transformed = clf_transformed.predict(X_test_transformed)

        # Evaluate fairness metrics
        # For original data
        X_test_orig_copy = X_test_orig.copy()
        X_test_orig_copy[self.TARGET] = y_pred_original
        orig_dataset = DataFrame(X_test_orig_copy)
        orig_dataset.targets = self.TARGET
        orig_dataset.sensitive = self.SENSITIVE
        spd_original = orig_dataset.statistical_parity_difference()
        di_original = orig_dataset.disparate_impact()
        
        # For transformed data
        X_test_orig_copy = X_test_orig.copy()
        X_test_orig_copy[self.TARGET] = y_pred_transformed
        trans_dataset = DataFrame(X_test_orig_copy)
        trans_dataset.targets = self.TARGET
        trans_dataset.sensitive = self.SENSITIVE
        spd_transformed = trans_dataset.statistical_parity_difference()
        di_transformed = trans_dataset.disparate_impact()

        # Check if fairness improved - use a tolerance factor for comparison
        tolerance = 0.05  # 5% tolerance for fairness metrics
        for key in spd_original:
            self.assertLessEqual(
                abs(spd_transformed[key]),
                abs(spd_original[key]) * (1 + tolerance),
                f"SPD for {key} should be improved with LFR",
            )
        for key in di_original:
            self.assertLessEqual(
                abs(di_transformed[key] - 1),
                abs(di_original[key] - 1) * (1 + tolerance),
                f"DI for {key} should be improved with LFR",
            )

    def tearDown(self):
        del self.df


if __name__ == "__main__":
    unittest.main()
