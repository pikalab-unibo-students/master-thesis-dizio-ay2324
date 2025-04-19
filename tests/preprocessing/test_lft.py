import unittest
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from fairlib.preprocessing.lfr import LFR
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
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf


class TestLFR(unittest.TestCase):
    def setUp(self):
        """
        Set up the test case by loading the dataset and initializing the input features and target labels.
        """
        np.random.seed(42)
        torch.manual_seed(42)
        dataset = biased_dataset_people_height(binary=True)
        self.X = dataset.drop(columns=["class"])
        self.y = dataset["class"]
        self.sensitive_attr = dataset["male"]

    def test_lfr_model(self):
        # Split data
        X_train, X_test, y_train, y_test, s_train, s_test = train_test_split(
            self.X, self.y, self.sensitive_attr, test_size=0.2, random_state=42
        )

        # Initialize and train LFR model
        lfr = LFR(
            input_dim=self.X.shape[1],
            latent_dim=8,
            output_dim=self.X.shape[1],
            alpha_z=1.0,
            alpha_x=1.0,
            alpha_y=1.0,
        )

        # Transform the data using LFR
        lfr.fit(X_train, y_train, s_train, epochs=100)
        X_train_transformed = lfr.transform(X_train)
        X_test_transformed = lfr.transform(X_test)

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

        for key in spd_original:
            self.assertLessEqual(
                abs(spd_transformed[key]),
                abs(spd_original[key]),
                "SPD should be improved with LFR",
            )
        for key in di_original:
            self.assertLessEqual(
                abs(di_transformed[key] - 1),
                abs(di_original[key] - 1),
                "DI should be improved with LFR",
            )
