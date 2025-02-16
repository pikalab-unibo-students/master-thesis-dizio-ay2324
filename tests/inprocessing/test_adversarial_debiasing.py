import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd

from fairlib import DataFrame
from tests.data_generator import biased_dataset_people_height
from fairlib.inprocessing.adversarial_debiasing import Predictor, Adversary, AdversarialDebiasingModel

def evaluate_fairness(X_test, y_pred):
    X_test = X_test.copy()
    X_test["class"] = y_pred
    dataset = DataFrame(X_test)
    dataset.targets = "class"
    dataset.sensitive = "male"
    spd = dataset.statistical_parity_difference()
    di = dataset.disparate_impact()
    return spd, di

def train_classifier(X_train, y_train):
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    return clf

class TestAdversarialDebiasingModel(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        dataset = biased_dataset_people_height(size=50000, binary=True)
        self.df = dataset
        self.X = dataset[["height", "age", "weight", "income"]].values
        self.y = dataset["class"].values
        self.a = dataset["male"].values

    def test_adversarial_debiasing(self):
        X_train, X_test, y_train, y_test, a_train, a_test = train_test_split(
            self.X, self.y, self.a, test_size=0.3, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_train_tensor = torch.from_numpy(X_train_scaled.astype(np.float32))
        y_train_tensor = torch.from_numpy(y_train.astype(np.int64))
        a_train_tensor = torch.from_numpy(a_train.astype(np.int64))
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, a_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=120, shuffle=True)

        clf = train_classifier(X_train_scaled, y_train)
        y_pred_baseline = clf.predict(X_test_scaled)

        X_test_df = pd.DataFrame(
            X_test_scaled,
            columns=["height", "age", "weight", "income"]
        )
        X_test_df["male"] = a_test

        spd_baseline, di_baseline = evaluate_fairness(X_test_df.copy(), y_pred_baseline)

        predictor = Predictor(input_dim=4, hidden_dim=16, output_dim=2)
        adversary = Adversary(input_dim=16, hidden_dim=16, sensitive_dim=1)
        model = AdversarialDebiasingModel(predictor, adversary, lambda_adv=5.0)
        model.fit(train_loader, num_epochs=50, lr=0.0001)

        model.eval()
        X_test_tensor = torch.from_numpy(X_test_scaled.astype(np.float32))
        with torch.no_grad():
            logits = model(X_test_tensor)
            y_pred_tensor = torch.argmax(logits, dim=1)
        y_pred_debiased = y_pred_tensor.cpu().numpy()

        spd_debiased, di_debiased = evaluate_fairness(X_test_df.copy(), y_pred_debiased)

        print("Baseline: SPD:", spd_baseline)
        print("Baseline: DI:", di_baseline)
        print("Adversarial Debiasing: SPD:", spd_debiased)
        print("Adversarial Debiasing: DI:", di_debiased)

        for key in spd_baseline:
            self.assertLessEqual(
                abs(spd_debiased[key]),
                abs(spd_baseline[key]),
                f"For Group: {key}, SPD should improve with Adversarial Debiasing"
            )
        for key in di_baseline:
            self.assertLessEqual(
                abs(di_debiased[key] - 1),
                abs(di_baseline[key] - 1),
                f"For Group: {key}, DI should improve with Adversarial Debiasing"
            )

if __name__ == '__main__':
    unittest.main()
