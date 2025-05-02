from typing_extensions import override
import torch
import torch.nn as nn
import torch.optim as optim
from fairlib import DataFrame, logger
from sklearn.preprocessing import StandardScaler
from typing import Optional, Any

from fairlib import logger
from fairlib.processing import (
    DataFrameAwareEstimator,
    DataFrameAwarePredictor,
    DataFrameAwareTransformer,
)


class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
        """
        Encoder network to transform input features into fair representations

        Parameters:
        -----------
        input_dim: int
            Dimension of input features
        latent_dim: int
            Dimension of the learned fair representation
        """
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, output_dim):
        """
        Decoder network to reconstruct original features from fair representations
        """
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, z):
        return self.decoder(z)


class Classifier(nn.Module):
    def __init__(self, latent_dim):
        """
        Classifier network to predict target variable from fair representations
        """
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, z):
        return self.classifier(z)


def compute_fairness_loss(z, sensitive_attr):
    """
    Compute statistical parity loss (Lz)
    """
    protected_mask = sensitive_attr == 1
    unprotected_mask = sensitive_attr == 0

    protected_mean = torch.mean(z[protected_mask], dim=0)
    unprotected_mean = torch.mean(z[unprotected_mask], dim=0)

    return torch.sum((protected_mean - unprotected_mean) ** 2)


def compute_reconstruction_loss(x, x_reconstructed):
    """
    Compute reconstruction loss (Lx)
    """
    return torch.mean((x - x_reconstructed) ** 2)


def compute_classification_loss(y_pred, y_true):
    """
    Compute binary cross-entropy loss (Ly)
    """
    return nn.BCELoss()(y_pred, y_true)


class LFR(DataFrameAwareEstimator, DataFrameAwarePredictor, DataFrameAwareTransformer):

    def __init__(self,
                 input_dim = None,
                 latent_dim = None,
                 output_dim = None,
                 encoder = None,
                 decoder = None,
                 classifier = None,
                 alpha_z=1.0,
                 alpha_x=1.0,
                 alpha_y=1.0):
        """
        Learning Fair Representations (LFR) model

        Parameters:
        -----------
        input_dim: int
            Dimension of input features
        latent_dim: int
            Dimension of the learned fair representation
        output_dim: int
            Dimension of the output features
        encoder: nn.Module
            Encoder network
        decoder: nn.Module
            Decoder network
        alpha_z: float
            Weight for the fairness loss (Lz)
        alpha_x: float
            Weight for the reconstruction loss (Lx)
        alpha_y: float
            Weight for the classification loss (Ly)
        """
        if encoder is None and decoder is None and classifier is None:
            if input_dim is None or latent_dim is None or output_dim is None:
                raise ValueError("input_dim, latent_dim, and output_dim must be provided")
            self.encoder = Encoder(input_dim, latent_dim)
            self.decoder = Decoder(latent_dim, output_dim)
            self.classifier = Classifier(latent_dim)
        elif encoder is None or decoder is None or classifier is None:
            raise ValueError("Both encoder and decoder must be provided or None")
        else:
            self.encoder = encoder
            self.decoder = decoder
            self.classifier = classifier

        self.alpha_z = alpha_z
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

        # Initialize scaler
        self.scaler = StandardScaler()

    @override
    def fit(self, 
            X,
            y: Optional[Any] = None,
            epochs=100, 
            batch_size=32, 
            learning_rate=0.001
            ):
        """
        Train the LFR model
        """
        # Prepare data
        if not isinstance(X, DataFrame):
            raise TypeError(f"Expected a DataFrame, got {type(X)}")
        if y is None:
            X, y, _, _, _, sensitive_indexes = X.unpack()
        else:
            X, _, _, _, _, sensitive_indexes = X.unpack()
        if len(sensitive_indexes) != 1:
            raise ValueError(
                f"LFR expects exactly one sensitive column, got {len(sensitive_indexes)}: {X.sensitive}"
            )
        sensitive_attr = X[:, sensitive_indexes[0]]
        X = self.scaler.fit_transform(X)
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).reshape(-1, 1)
        sensitive_attr = torch.FloatTensor(sensitive_attr)

        # Create optimizer
        optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.classifier.parameters()),
            lr=learning_rate,
        )

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            z = self.encoder(X)
            x_reconstructed = self.decoder(z)
            y_pred = self.classifier(z)

            # Compute losses
            fairness_loss = compute_fairness_loss(z, sensitive_attr)
            reconstruction_loss = compute_reconstruction_loss(X, x_reconstructed)
            classification_loss = compute_classification_loss(y_pred, y)

            # Compute total loss
            total_loss = (
                self.alpha_z * fairness_loss
                + self.alpha_x * reconstruction_loss
                + self.alpha_y * classification_loss
            )

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Loss: {total_loss.item():.4f}, "
                    f"Fairness: {fairness_loss.item():.4f}, "
                    f"Reconstruction: {reconstruction_loss.item():.4f}, "
                    f"Classification: {classification_loss.item():.4f}"
                )

    def _predict(self, X):
        """
        Make predictions on new data
        """
        X = self.scaler.transform(X)
        X = torch.FloatTensor(X)
        z = self.encoder(X)
        y_pred = self.classifier(z)
        return (y_pred.detach().numpy() > 0.5).astype(int)

    def _transform(self, X, y=None):
        """
        Transform data into fair representations
        """
        X = self.scaler.transform(X)
        X = torch.FloatTensor(X)
        z = self.encoder(X)
        return z.detach().numpy()
