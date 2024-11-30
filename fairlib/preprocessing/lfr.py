import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from fairlib.preprocessing._lfr_model import Encoder, Decoder, Classifier
from fairlib.dataframe import DataFrame


def compute_reconstruction_loss(x, x_reconstructed):
    return torch.mean((x - x_reconstructed) ** 2)


def compute_fairness_loss(z, sensitive_attr):
    protected_mask = sensitive_attr == 1
    unprotected_mask = sensitive_attr == 0

    protected_mean = torch.mean(z[protected_mask], dim=0)
    unprotected_mean = torch.mean(z[unprotected_mask], dim=0)

    return torch.sum((protected_mean - unprotected_mean) ** 2)


def compute_classification_loss(y_pred, y_true):
    return nn.BCELoss()(y_pred, y_true)


class LFR:
    def __init__(self, input_dim, latent_dim, alpha_z=1.0, alpha_x=1.0, alpha_y=1.0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.alpha_z = alpha_z
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.classifier = Classifier(latent_dim)

        self.scaler = StandardScaler()

    def fit(self, df: DataFrame, epochs=100, batch_size=32, learning_rate=0.001):

        if len(df.targets) > 1:
            raise ValueError(
                "More than one “target” column is present. LFR supports only 1 target."
            )
        target_columns = df.targets.pop()
        if len(df.sensitive) > 1:
            raise ValueError(
                "More than one “sensitive” column is present. LFR supports only 1 sensitive."
            )
        sensitive_columns = df.sensitive.pop()

        X = df.drop(columns=target_columns).values
        y = df[target_columns].values
        sensitive_attr = df[sensitive_columns].values

        X = self.scaler.fit_transform(X)
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).reshape(-1, 1)
        sensitive_attr = torch.FloatTensor(sensitive_attr)

        optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.classifier.parameters()),
            lr=learning_rate,
        )

        for epoch in range(epochs):
            z = self.encoder(X)
            x_reconstructed = self.decoder(z)
            y_pred = self.classifier(z)

            fairness_loss = compute_fairness_loss(z, sensitive_attr)
            reconstruction_loss = compute_reconstruction_loss(X, x_reconstructed)
            classification_loss = compute_classification_loss(y_pred, y)

            total_loss = (
                self.alpha_z * fairness_loss
                + self.alpha_x * reconstruction_loss
                + self.alpha_y * classification_loss
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Loss: {total_loss.item():.4f}, "
                    f"Fairness: {fairness_loss.item():.4f}, "
                    f"Reconstruction: {reconstruction_loss.item():.4f}, "
                    f"Classification: {classification_loss.item():.4f}"
                )

    def predict(self, df: DataFrame):
        X = df.drop(columns=df.targets).values
        X = self.scaler.transform(X)
        X = torch.FloatTensor(X)
        z = self.encoder(X)
        y_pred = self.classifier(z)
        return (y_pred.detach().numpy() > 0.5).astype(int)

    def transform(self, df: DataFrame):
        X = df.drop(columns=df.targets).values
        X = self.scaler.transform(X)
        X = torch.FloatTensor(X)
        z = self.encoder(X)
        return z.detach().numpy()


def main():
    import fairlib as fl

    df = fl.DataFrame(
        {
            "name": ["Alice", "Bob", "Carla", "Davide", "Elena"],
            "age": [25, 32, 45, 29, 34],
            "sex": ["F", "M", "F", "M", "F"],
            "income": [40000, 50000, 45000, 53000, 43000],
        }
    )

    targe_column = "income"
    sensitive_column = "sex"

    df.targets = targe_column
    df.sensitive = sensitive_column

    # Drop name column
    df = df.drop(columns="name")

    # Make sex column binary
    df[sensitive_column] = df[sensitive_column].apply(lambda x: x == "M").astype(int)
    df[targe_column] = df[targe_column].apply(lambda x: x > 50000).astype(int)

    input_dim = df.shape[1] - 1

    lfr = LFR(input_dim=input_dim, latent_dim=2)
    lfr.fit(df, epochs=100)
    predictions = lfr.predict(df)
    print(predictions)


if __name__ == "__main__":
    main()
