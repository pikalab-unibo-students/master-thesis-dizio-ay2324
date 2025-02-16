import unittest
import fairlib as fl
from fairlib.inprocessing import Fauci
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
        input_shape (int): The number of input features.non

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
    Prepare the data for training the Fauci model.

    Args:
        X (DataFrame): The input features.
        y (Series): The target labels.
        target (str): The name of the target column.
        sensitive (str): The name of the sensitive attribute column.

    Returns:
        tuple: A tuple containing the prepared Fauci dataset, input features, and target labels.
    """
    X_train = X.copy()
    y_train = y.copy()
    X_train[target] = y_train
    fauci_train_dataset = fl.DataFrame(X_train)
    fauci_train_dataset.targets = target
    fauci_train_dataset.sensitive = sensitive
    X_train.drop(columns=[target], inplace=True)
    return fauci_train_dataset, X_train, y_train


def evaluate_model(model: Fauci, X_test, y_test, typeOfEval, target, sensitive):
    """
    Evaluate the model's accuracy and fairness metric.

    Args:
        model (nn.Module): The trained model.
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


class TestFauci(unittest.TestCase):
    """
    Test class for the Fauci fairness-aware learning model.
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
        # Load dataset from OpenML
        dataset = biased_dataset_people_height(binary=True)

        self.X = fl.DataFrame(dataset)
        self.y = self.X.pop(self.TARGET)
        self.num_features = self.X.shape[1]

    def train_and_evaluate(self, regularizer, fairness_metric):
        """
        Train and evaluate the Fauci model and a default model.

        Args:
            regularizer (str): The type of fairness regularization to use.
            fairness_metric (str): The type of fairness metric to evaluate.

        Returns:
            tuple: A tuple containing the accuracy and fairness metric for both the Fauci and default models.
        """
        # Prepare data for Fauci
        fauci_train_dataset, X_train, y_train = get_prepared_data(
            self.X, self.y, self.TARGET, self.SENSITIVE
        )

        # Train Fauci model
        fauci_model = create_model(self.num_features)
        fauciModel = Fauci(
            fauci_model,
            loss=nn.BCELoss(),
            fairness_regularization=regularizer,
            regularization_weight=0.6,
        )
        fauciModel.fit(
            fauci_train_dataset,
            epochs=self.EPOCHS,
            batch_size=self.BATCH_SIZE,
        )

        # Evaluate Fauci model
        fauci_accuracy, fauci_metric = evaluate_model(
            fauciModel, X_train, y_train, fairness_metric, self.TARGET, self.SENSITIVE
        )

        fauci_train_dataset, X_train, y_train = get_prepared_data(
            self.X, self.y, self.TARGET, self.SENSITIVE
        )

        # Train default model
        default_model = create_model(self.num_features)
        default_model = Fauci(
            default_model,
            loss=nn.BCELoss(),
            fairness_regularization=None,
            regularization_weight=0.0,
        )
        default_model.fit(
            fauci_train_dataset,
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

        return fauci_accuracy, fauci_metric, default_accuracy, default_metric

    def testFauciOneSensitiveAttrSPD(self):
        """
        Test the Fauci model using Statistical Parity Difference (SPD) as the fairness metric.
        """
        fauci_accuracy, fauci_spd, default_accuracy, default_spd = (
            self.train_and_evaluate(regularizer="sp", fairness_metric="spd")
        )


        # Assert fairness improvement
        fauci_spd_value = abs(fauci_spd[{self.TARGET: 1, self.SENSITIVE: 1}])
        default_spd_value = abs(default_spd[{self.TARGET: 1, self.SENSITIVE: 1}])

        assert (
            fauci_spd_value <= default_spd_value
        ), f"Expected {fauci_spd_value} to be less than {default_spd_value}"

    def testFauciOneSensitiveAttrDI(self):
        """
        Test the Fauci model using Disparate Impact (DI) as the fairness metric.
        """
        fauci_accuracy, fauci_di, default_accuracy, default_di = (
            self.train_and_evaluate(regularizer="di", fairness_metric="di")
        )

        # Assert fairness improvement
        fauci_distance = abs(fauci_di[{self.TARGET: 1, self.SENSITIVE: 1}] - 1)
        model_distance = abs(default_di[{self.TARGET: 1, self.SENSITIVE: 1}] - 1)
        assert (
            fauci_distance <= model_distance
        ), f"Expected {fauci_distance}, to be less than {model_distance}"


if __name__ == "__main__":
    unittest.main()
