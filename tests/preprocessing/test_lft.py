import unittest
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from fairlib.preprocessing.lfr import LFR
from fairlib import DataFrame
from tests.data_generator import biased_dataset_people_height


def evaluate_fairness(X_test, y_pred):
    """
    Evaluate the fairness metrics (SPD and DI) of the predictions.
    """
    X_test = X_test.copy()
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


def get_prepared_data(X, y, target, sensitive):
    """
    Prepare the data for training the LFR model.

    Args:
        X (DataFrame): The input features.
        y (Series): The target labels.
        target (str): The name of the target column.
        sensitive (str): The name of the sensitive attribute column.

    Returns:
        tuple: A tuple containing the prepared dataset, input features, and target labels.
    """
    X_train = X.copy()
    y_train = y.copy()
    X_train[target] = y_train
    train_dataset = DataFrame(X_train)
    train_dataset.targets = target
    train_dataset.sensitive = sensitive
    X_train.drop(columns=[target], inplace=True)
    return train_dataset, X_train, y_train


class TestLFR(unittest.TestCase):
    """
    Test class for the LFR fairness-aware learning model.
    """
    EPOCHS = 20
    BATCH_SIZE = 120
    TARGET = "class"
    SENSITIVE = "male"
    
    def setUp(self):
        """
        Set up the test case by loading the dataset and initializing the input features and target labels.
        """
        np.random.seed(42)
        torch.manual_seed(42)
        dataset = biased_dataset_people_height(binary=True)
        self.X = DataFrame(dataset)
        self.y = self.X.pop(self.TARGET)

    def test_lfr_model(self):
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )

        # Prepare data for LFR
        lfr_train_dataset, X_train_copy, y_train_copy = get_prepared_data(
            X_train, y_train, self.TARGET, self.SENSITIVE
        )

        # Initialize and train LFR model
        lfr = LFR(
            input_dim=X_train.shape[1],
            latent_dim=8,
            output_dim=X_train.shape[1],
            alpha_z=1.0,
            alpha_x=1.0,
            alpha_y=1.0,
        )

        # Transform the data using LFR
        lfr.fit(lfr_train_dataset, epochs=self.EPOCHS, batch_size=self.BATCH_SIZE)
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

        # Check if fairness improved
        for key in spd_original:
            self.assertLessEqual(
                abs(spd_transformed[key]),
                abs(spd_original[key]),
                f"SPD for {key} should be improved with LFR"
            )
        for key in di_original:
            self.assertLessEqual(
                abs(di_transformed[key] - 1),
                abs(di_original[key] - 1),
                f"DI for {key} should be improved with LFR"
            )



if __name__ == "__main__":
    unittest.main()
