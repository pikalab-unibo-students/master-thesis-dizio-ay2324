import numpy as np
import torch
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from fairlib import DataFrame


def set_seed(seed=42):
    """
    Set the seed for reproducibility across multiple libraries.

    Args:
        seed (int): The seed to set. Default is 42.
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


def train_classifier(X_train, y_train, random_state=42, max_iter=1000):
    """
    Train a simple classifier on the given data.

    Args:
        X_train: Training features
        y_train: Training labels
        random_state (int): Random state for reproducibility
        max_iter (int): Maximum number of iterations

    Returns:
        LogisticRegression: Trained classifier
    """
    clf = LogisticRegression(random_state=random_state, max_iter=max_iter)
    clf.fit(X_train, y_train)
    return clf


def evaluate_fairness(X_test, y_pred, target_col="class", sensitive_col="male"):
    """
    Evaluate the fairness metrics (SPD and DI) of the predictions.

    Args:
        X_test: Test features
        y_pred: Predicted labels
        target_col (str): Name of the target column
        sensitive_col (str): Name of the sensitive attribute column

    Returns:
        tuple: Statistical Parity Difference and Disparate Impact metrics
    """
    X_test = X_test.copy()
    X_test[target_col] = y_pred
    dataset = DataFrame(X_test)
    dataset.targets = target_col
    dataset.sensitive = sensitive_col

    spd = dataset.statistical_parity_difference()
    di = dataset.disparate_impact()
    return spd, di


def evaluate_model(model, X_test, y_test, fairness_metric, target_col, sensitive_col):
    """
    Evaluate the model's accuracy and fairness metric.

    Args:
        model: The trained model with a predict method
        X_test: Test features
        y_test: Test labels
        fairness_metric (str): The type of fairness metric to evaluate ('spd' or 'di')
        target_col (str): Name of the target column
        sensitive_col (str): Name of the sensitive attribute column

    Returns:
        tuple: Accuracy and fairness metric
    """
    # Handle different model types (neural network vs sklearn)
    if hasattr(model, "predict") and callable(getattr(model, "predict")):
        if (
            hasattr(model.predict, "__self__")
            and hasattr(model.predict.__self__, "__class__")
            and "torch" in model.predict.__self__.__class__.__module__
        ):
            # PyTorch model
            predictions = model.predict(X_test).detach().numpy()
            y_pred = (predictions > 0.5).astype(int)
        else:
            # Sklearn model
            y_pred = model.predict(X_test)
    else:
        raise ValueError("Model must have a predict method")

    accuracy = accuracy_score(y_test, y_pred)

    # Convert test set to FairLib DataFrame
    X_test_copy = X_test.copy()
    X_test_copy[target_col] = y_pred
    dataset = DataFrame(X_test_copy)
    dataset.targets = target_col
    dataset.sensitive = sensitive_col

    # Evaluate fairness metric
    if fairness_metric == "spd":
        metric = dataset.statistical_parity_difference()
    elif fairness_metric == "di":
        metric = dataset.disparate_impact()
    else:
        raise ValueError(f"Unknown fairness metric: {fairness_metric}")

    return accuracy, metric


def get_prepared_data(X, y, target_col, sensitive_col):
    """
    Prepare the data for training fairness-aware models.

    Args:
        X: Input features
        y: Target labels
        target_col (str): Name of the target column
        sensitive_col (str): Name of the sensitive attribute column

    Returns:
        tuple: A tuple containing the prepared dataset, input features, and target labels
    """
    X_train = X.copy()
    y_train = y.copy()
    X_train[target_col] = y_train
    train_dataset = DataFrame(X_train)
    train_dataset.targets = target_col
    train_dataset.sensitive = sensitive_col
    X_train_features = X_train.copy()
    if target_col in X_train_features.columns:
        X_train_features.drop(columns=[target_col], inplace=True)
    return train_dataset, X_train_features, y_train
