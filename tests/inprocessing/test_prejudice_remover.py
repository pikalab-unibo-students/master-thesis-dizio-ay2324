import unittest
import fairlib as fl
from fairlib.inprocessing.prejudice_remover import PrejudiceRemover
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
import random
import numpy as np
from tests.data_generator import biased_dataset_people_height


def set_seed(seed):
    """
    Set the seed for reproducibility across multiple libraries.

    Args:
        seed (int): The seed to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(input_shape):
    """
    Create a basic neural network model with the specified input shape.

    Args:
        input_shape (int): The number of input features.

    Returns:
        nn.Sequential: A neural network model.
    """
    net = nn.Sequential(
        nn.Linear(input_shape, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 16),
        nn.ReLU(),
        nn.Linear(16, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
        nn.Sigmoid(),
    )

    return net


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


def evaluate_model(model, X_test, y_test, typeOfEval, target, sensitive):
    """
    Evaluate the model's accuracy and fairness metric.

    Args:
        model: The trained model.
        X_test (DataFrame): The test input features.
        y_test (Series): The test target labels.
        typeOfEval (str): The type of fairness metric to evaluate ('spd' or 'di').
        target (str): The name of the target column.
        sensitive (str): The name of the sensitive attribute column.

    Returns:
        tuple: A tuple containing the accuracy and the fairness metric.
    """
    predictions = model.predict(X_test).detach().numpy()
    y_pred = (predictions > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)

    # Convert test set to FairLib DataFrame
    X_test[target] = y_pred
    dataset = fl.DataFrame(X_test)
    dataset.targets = target
    dataset.sensitive = sensitive

    # Evaluate fairness metric
    if typeOfEval == "spd":
        metric = dataset.statistical_parity_difference()
    elif typeOfEval == "di":
        metric = dataset.disparate_impact()
    else:
        raise ValueError("Metric Not Found")
    return accuracy, metric


class TestPrejudiceRemover(unittest.TestCase):
    """
    Test class for the PrejudiceRemover fairness-aware learning model.
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
        # Load dataset
        dataset = biased_dataset_people_height(binary=True)

        self.X = fl.DataFrame(dataset)
        self.y = self.X.pop(self.TARGET)
        self.num_features = self.X.shape[1]

    def train_and_evaluate(self, eta, fairness_metric):
        """
        Train and evaluate the PrejudiceRemover model and a default model.

        Args:
            eta (float): The fairness regularization parameter.
            fairness_metric (str): The type of fairness metric to evaluate.

        Returns:
            tuple: A tuple containing the accuracy and fairness metric for both models.
        """
        # Prepare data
        train_dataset, X_train, y_train = get_prepared_data(
            self.X, self.y, self.TARGET, self.SENSITIVE
        )

        # Train PrejudiceRemover model
        pr_model = create_model(self.num_features)
        prejudice_remover = PrejudiceRemover(
            pr_model,
            loss=nn.BCELoss(),
            eta=eta,
        )
        prejudice_remover.fit(
            train_dataset,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
        )

        # Evaluate PrejudiceRemover model
        pr_accuracy, pr_metric = evaluate_model(
            prejudice_remover,
            X_train,
            y_train,
            fairness_metric,
            self.TARGET,
            self.SENSITIVE,
        )

        # Prepare data for default model
        train_dataset, X_train, y_train = get_prepared_data(
            self.X, self.y, self.TARGET, self.SENSITIVE
        )

        # Train default model (no fairness regularization)
        default_model = create_model(self.num_features)
        default_model = PrejudiceRemover(
            default_model,
            loss=nn.BCELoss(),
            eta=0.0,  # No fairness regularization
        )
        default_model.fit(
            train_dataset,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
        )

        # Evaluate default model
        default_accuracy, default_metric = evaluate_model(
            default_model,
            X_train,
            y_train,
            fairness_metric,
            self.TARGET,
            self.SENSITIVE,
        )

        return pr_accuracy, pr_metric, default_accuracy, default_metric

    def testPrejudiceRemoverSPD(self):
        """
        Test the PrejudiceRemover model using Statistical Parity Difference (SPD) as the fairness metric.
        """
        _, pr_spd, _, default_spd = self.train_and_evaluate(
            eta=1.0, fairness_metric="spd"
        )

        # Assert fairness improvement
        pr_spd_value = abs(pr_spd[{self.TARGET: 1, self.SENSITIVE: 1}])
        default_spd_value = abs(default_spd[{self.TARGET: 1, self.SENSITIVE: 1}])

        self.assertLessEqual(
            pr_spd_value,
            default_spd_value,
            f"Expected {pr_spd_value} to be less than or equal to {default_spd_value}",
        )

    def testPrejudiceRemoverDI(self):
        """
        Test the PrejudiceRemover model using Disparate Impact (DI) as the fairness metric.
        """
        _, pr_di, _, default_di = self.train_and_evaluate(eta=1.0, fairness_metric="di")

        # Assert fairness improvement
        pr_distance = abs(pr_di[{self.TARGET: 1, self.SENSITIVE: 1}] - 1)
        default_distance = abs(default_di[{self.TARGET: 1, self.SENSITIVE: 1}] - 1)

        self.assertLessEqual(
            pr_distance,
            default_distance,
            f"Expected {pr_distance} to be less than or equal to {default_distance}",
        )


if __name__ == "__main__":
    unittest.main()
