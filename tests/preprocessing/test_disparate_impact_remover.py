import unittest
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fairlib.preprocessing.disparate_impact_remover import DisparateImpactRemover
from fairlib import DataFrame
from tests.data_generator import biased_dataset_people_height


def evaluate_fairness(X_test, y_pred):
    """
    Evaluate the fairness metrics (SPD and DI) of the predictions.
    """
    X_test["class"] = y_pred
    dataset = DataFrame(X_test)
    dataset.targets = "class"
    dataset.sensitive = "male"

    spd = dataset.statistical_parity_difference()
    di = dataset.disparate_impact()
    return spd, di


def train_classifier(X_train, y_train):
    """
    Train a simple classifier on the given data.
    """
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_train, y_train)
    return clf


class TestDisparateImpactRemover(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case by loading the dataset and initializing the input features and target labels.
        """
        np.random.seed(42)
        dataset = biased_dataset_people_height(binary=True)
        self.X = dataset.drop(columns=["class"])
        self.y = dataset["class"]
        self.sensitive_attr = dataset["male"]
        
    def test_disparate_impact_remover_model(self):
        # Split data
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            self.X, self.y, self.sensitive_attr, test_size=0.2, random_state=42
        )

        # Initialize and train DisparateImpactRemover model
        dir_model = DisparateImpactRemover(repair_level=0.8)

        # Transform the data using DisparateImpactRemover
        X_train_transformed = dir_model.fit_transform(X_train, y_train, s=s_train)
        X_test_transformed = dir_model.transform(X_test, s=s_test)

        # Train classifier on original data
        clf_original = train_classifier(X_train, y_train)
        y_pred_original = clf_original.predict(X_test)

        # Train classifier on transformed data
        clf_transformed = train_classifier(X_train_transformed, y_train)
        y_pred_transformed = clf_transformed.predict(X_test_transformed)

        # Evaluate fairness metrics
        spd_original, di_original = evaluate_fairness(X_test.copy(), y_pred_original)
        spd_transformed, di_transformed = evaluate_fairness(
            X_test.copy(), y_pred_transformed
        )

        # Check if fairness improved
        for key in spd_original:
            self.assertLessEqual(
                abs(spd_transformed[key]),
                abs(spd_original[key]),
                f"SPD for {key} should be improved with DisparateImpactRemover"
            )
        for key in di_original:
            self.assertLessEqual(
                abs(di_transformed[key] - 1),
                abs(di_original[key] - 1),
                f"DI for {key} should be improved with DisparateImpactRemover"
            )
            
    def test_all_data_transformation(self):
        """Test that all data can be transformed correctly."""
        dir_model = DisparateImpactRemover(repair_level=0.5)
        
        dir_model.fit(self.X, self.y, s=self.sensitive_attr)
        X_transformed = dir_model.transform(self.X, s=self.sensitive_attr)
        
        # Check output shape
        self.assertEqual(X_transformed.shape, self.X.shape)
        
        # Check that values are different after transformation
        self.assertFalse(np.array_equal(X_transformed.values, self.X.values))
        
    def test_repair_level_effect(self):
        """Test that repair_level affects the transformation."""
        # Two models with different repair levels
        dir_no_repair = DisparateImpactRemover(repair_level=0)
        dir_full_repair = DisparateImpactRemover(repair_level=1)
        
        dir_no_repair.fit(self.X, self.y, s=self.sensitive_attr)
        dir_full_repair.fit(self.X, self.y, s=self.sensitive_attr)
        
        X_no_repair = dir_no_repair.transform(self.X, s=self.sensitive_attr)
        X_full_repair = dir_full_repair.transform(self.X, s=self.sensitive_attr)
        
        # With repair_level=0, the output should be almost identical to input
        np.testing.assert_array_almost_equal(X_no_repair.values, self.X.values, decimal=5)
        
        # With repair_level=1, the output should be different from the input
        self.assertFalse(np.array_equal(X_full_repair.values, self.X.values))


if __name__ == "__main__":
    unittest.main()