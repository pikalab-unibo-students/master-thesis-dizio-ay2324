import unittest
import fairlib as fl
from fairlib.inprocessing.adversarial_debiasing import (
    Predictor,
    Adversary,
    AdversarialDebiasingModel,
)
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import random
import numpy as np
from tests.data_generator import biased_dataset_people_height


def set_seed(seed):
    """
    Set the seed for reproducibility across multiple libraries.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CustomPredictor(nn.Module):
    """
    Custom predictor model that supports returning intermediate representations.
    This is needed for compatibility with the AdversarialDebiasingModel.
    """

    def __init__(self, input_shape):
        super(CustomPredictor, self).__init__()
        self.fc1 = nn.Linear(input_shape, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(16, 8)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(8, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_representation=False):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.relu3(self.fc3(x))
        rep = self.relu4(
            self.fc4(x)
        )  # This is the representation we'll use for adversarial training
        output = self.sigmoid(self.fc5(rep))

        if return_representation:
            return output, rep
        return output


def create_model(input_shape):
    """
    Create a custom neural network model with the specified input shape.

    Args:
        input_shape (int): The number of input features.

    Returns:
        CustomPredictor: A neural network model compatible with adversarial debiasing.
    """
    return CustomPredictor(input_shape)


def get_prepared_data(X, y, target, sensitive):
    """
    Prepare the data for training the model.

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
    train_dataset = fl.DataFrame(X_train)
    train_dataset.targets = target
    train_dataset.sensitive = sensitive
    X_train.drop(columns=[target], inplace=True)
    return train_dataset, X_train, y_train


def evaluate_model(model, X_test, y_test, fairness_metric, target, sensitive):
    """
    Evaluate the model's accuracy and fairness metric.

    Args:
        model: The trained model.
        X_test (DataFrame): The test input features.
        y_test (Series): The test target labels.
        fairness_metric (str): The type of fairness metric to evaluate ('spd' or 'di').
        target (str): The name of the target column.
        sensitive (str): The name of the sensitive attribute column.

    Returns:
        tuple: A tuple containing the accuracy and the fairness metric.
    """
    # Convert input to tensor for prediction
    predictions = model.predict(X_test).detach().numpy()
    y_pred = (predictions > 0.5).astype(int).flatten()
    accuracy = accuracy_score(y_test, y_pred)

    # Convert test set to FairLib DataFrame
    X_test_copy = X_test.copy()
    X_test_copy[target] = y_pred
    dataset = fl.DataFrame(X_test_copy)
    dataset.targets = target
    dataset.sensitive = sensitive

    # Evaluate fairness metric
    if fairness_metric == "spd":
        metric = dataset.statistical_parity_difference()
    elif fairness_metric == "di":
        metric = dataset.disparate_impact()
    else:
        raise ValueError("Metric Not Found")
    return accuracy, metric


class TestAdversarialDebiasingModel(unittest.TestCase):
    """
    Test class for the Adversarial Debiasing model.
    """

    EPOCHS = 20
    BATCH_SIZE = 120
    TARGET = "class"
    SENSITIVE = "male"

    def setUp(self):
        """
        Set up the test case by loading the dataset and initializing the input features and target labels.
        """
        set_seed(42)
        dataset = biased_dataset_people_height(binary=True)
        self.X = fl.DataFrame(dataset)
        self.y = self.X.pop(self.TARGET)
        self.num_features = self.X.shape[1]

    def train_and_evaluate(self, lambda_adv, fairness_metric):
        """
        Train and evaluate the Adversarial Debiasing model and a baseline model.

        Args:
            lambda_adv (float): The weight of the adversarial loss.
            fairness_metric (str): The type of fairness metric to evaluate.

        Returns:
            tuple: A tuple containing the accuracy and fairness metric for both models.
        """
        # Prepare data
        train_dataset, X_train, y_train = get_prepared_data(
            self.X, self.y, self.TARGET, self.SENSITIVE
        )

        # Create model with the same architecture as other tests but compatible with adversarial debiasing
        adv_base_model = create_model(self.num_features)

        # The hidden representation dimension is 8 (from the second-to-last layer)
        hidden_dim = 8

        # Create adversary for the hidden representation
        adversary = Adversary(
            input_dim=hidden_dim, hidden_dim=hidden_dim, sensitive_dim=1
        )

        # Create wrapper for the model and adversary
        adv_model = AdversarialDebiasingModel(
            predictor=adv_base_model, adversary=adversary, lambda_adv=lambda_adv
        )

        # Train the adversarial model
        adv_model.fit(train_dataset, num_epochs=self.EPOCHS, batch_size=self.BATCH_SIZE)

        # Evaluate adversarial model
        adv_accuracy, adv_metric = evaluate_model(
            adv_model, X_train, y_train, fairness_metric, self.TARGET, self.SENSITIVE
        )

        # Prepare data for baseline model
        train_dataset, X_train, y_train = get_prepared_data(
            self.X, self.y, self.TARGET, self.SENSITIVE
        )

        # Train baseline model (no adversarial debiasing)
        baseline_predictor = create_model(self.num_features)
        # Create a dummy adversary (won't be used with lambda_adv=0.0)
        baseline_adversary = Adversary(
            input_dim=hidden_dim, hidden_dim=hidden_dim, sensitive_dim=1
        )
        baseline_model = AdversarialDebiasingModel(
            predictor=baseline_predictor,
            adversary=baseline_adversary,
            lambda_adv=0.0,  # No adversarial debiasing effect
        )
        baseline_model.fit(
            train_dataset, num_epochs=self.EPOCHS, batch_size=self.BATCH_SIZE
        )

        # Evaluate baseline model
        baseline_accuracy, baseline_metric = evaluate_model(
            baseline_model,
            X_train,
            y_train,
            fairness_metric,
            self.TARGET,
            self.SENSITIVE,
        )

        return adv_accuracy, adv_metric, baseline_accuracy, baseline_metric

    def test_adversarial_debiasing_spd(self):
        """
        Test the Adversarial Debiasing model using Statistical Parity Difference (SPD) as the fairness metric.
        """
        _, adv_spd, _, baseline_spd = self.train_and_evaluate(
            lambda_adv=5.0, fairness_metric="spd"
        )

        # Assert fairness improvement
        # Find the first key in the dictionary and use it to access the value
        adv_key = list(adv_spd.keys())[0]
        baseline_key = list(baseline_spd.keys())[0]

        # Extract the numeric values
        adv_spd_value = abs(adv_spd[adv_key])
        baseline_spd_value = abs(baseline_spd[baseline_key])

        self.assertLessEqual(
            adv_spd_value,
            baseline_spd_value,
            f"Expected {adv_spd_value} to be less than or equal to {baseline_spd_value}",
        )

    def test_adversarial_debiasing_di(self):
        """
        Test the Adversarial Debiasing model using Disparate Impact (DI) as the fairness metric.
        """
        _, adv_di, _, baseline_di = self.train_and_evaluate(
            lambda_adv=5.0, fairness_metric="di"
        )

        # Assert fairness improvement
        # Find the first key in the dictionary and use it to access the value
        adv_key = list(adv_di.keys())[0]
        baseline_key = list(baseline_di.keys())[0]

        # Extract the numeric values and calculate distance from 1 (ideal fairness)
        adv_distance = abs(adv_di[adv_key] - 1)
        baseline_distance = abs(baseline_di[baseline_key] - 1)

        self.assertLessEqual(
            adv_distance,
            baseline_distance,
            f"Expected {adv_distance} to be less than or equal to {baseline_distance}",
        )


if __name__ == "__main__":
    unittest.main()
