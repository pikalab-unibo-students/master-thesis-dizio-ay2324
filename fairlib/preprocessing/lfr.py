import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from fairlib import DataFrame
from sklearn.preprocessing import StandardScaler
from typing import Optional, Any, Tuple, List, Dict, Union, BinaryIO
import pickle
import os
from .pre_processing import Preprocessor
from .utils import validate_dataframe
from ..logging import logger


class Encoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        """
        Encoder network to transform input features into fair representations.

        Parameters
        ----------
        input_dim : int
            Dimension of input features
        latent_dim : int
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the encoder.

        Parameters
        ----------
        x : torch.Tensor
            Input features

        Returns
        -------
        torch.Tensor
            Encoded fair representation
        """
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, output_dim: int):
        """
        Decoder network to reconstruct original features from fair representations.

        Parameters
        ----------
        latent_dim : int
            Dimension of the fair representation
        output_dim : int
            Dimension of the output features (original feature space)
        """
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.

        Parameters
        ----------
        z : torch.Tensor
            Encoded fair representation

        Returns
        -------
        torch.Tensor
            Reconstructed features in original space
        """
        return self.decoder(z)


class Classifier(nn.Module):
    def __init__(self, latent_dim: int):
        """
        Classifier network to predict target variable from fair representations.

        Parameters
        ----------
        latent_dim : int
            Dimension of the fair representation
        """
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, 1), nn.Sigmoid()
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.

        Parameters
        ----------
        z : torch.Tensor
            Encoded fair representation

        Returns
        -------
        torch.Tensor
            Predicted probability for the target variable
        """
        return self.classifier(z)


def compute_fairness_loss(
    z: torch.Tensor, sensitive_attr: torch.Tensor
) -> torch.Tensor:
    """
    Compute statistical parity loss (Lz) to measure fairness.

    This function calculates the statistical parity loss by measuring the squared
    difference between the mean representations of protected and unprotected groups.
    Minimizing this loss helps achieve statistical parity (independence between
    sensitive attributes and model outputs).

    Parameters
    ----------
    z : torch.Tensor
        Encoded representations from the encoder network, shape (n_samples, latent_dim)
    sensitive_attr : torch.Tensor
        Binary sensitive attribute values (0 for unprotected, 1 for protected),
        shape (n_samples,)

    Returns
    -------
    torch.Tensor
        Fairness loss value (scalar tensor)

    Notes
    -----
    The loss is calculated as the sum of squared differences between the mean
    representations of the protected and unprotected groups across all dimensions.
    Lower values indicate better statistical parity.
    """
    # Create masks for protected and unprotected groups
    protected_mask = sensitive_attr == 1
    unprotected_mask = sensitive_attr == 0

    # Calculate mean representations for each group
    protected_mean = torch.mean(z[protected_mask], dim=0)
    unprotected_mean = torch.mean(z[unprotected_mask], dim=0)

    # Return the sum of squared differences between means
    return torch.sum((protected_mean - unprotected_mean) ** 2)


def compute_reconstruction_loss(
    x: torch.Tensor, x_reconstructed: torch.Tensor
) -> torch.Tensor:
    """
    Compute reconstruction loss (Lx) to measure information preservation.

    This function calculates the mean squared error between the original input
    features and their reconstructed versions from the decoder. Minimizing this
    loss ensures that the fair representations retain useful information from
    the original features.

    Parameters
    ----------
    x : torch.Tensor
        Original input features, shape (n_samples, n_features)
    x_reconstructed : torch.Tensor
        Reconstructed features from the decoder, shape (n_samples, n_features)

    Returns
    -------
    torch.Tensor
        Reconstruction loss value (scalar tensor)

    Notes
    -----
    The loss is calculated as the mean squared error (MSE) between original and
    reconstructed features. Lower values indicate better reconstruction quality
    and information preservation.
    """
    # Calculate mean squared error between original and reconstructed features
    return torch.mean((x - x_reconstructed) ** 2)


def compute_classification_loss(
    y_pred: torch.Tensor, y_true: torch.Tensor
) -> torch.Tensor:
    """
    Compute binary cross-entropy loss (Ly) for classification accuracy.

    This function calculates the binary cross-entropy loss between predicted and
    true target values. Minimizing this loss ensures that the fair representations
    maintain predictive power for the target variable.

    Parameters
    ----------
    y_pred : torch.Tensor
        Predicted target probabilities from the classifier, shape (n_samples, 1)
    y_true : torch.Tensor
        True binary target values, shape (n_samples, 1)

    Returns
    -------
    torch.Tensor
        Classification loss value (scalar tensor)

    Notes
    -----
    The loss is calculated using PyTorch's BCELoss (Binary Cross Entropy Loss).
    Lower values indicate better classification performance.

    This function assumes that y_pred contains valid probabilities (0-1 range)
    and that y_true contains binary values (0 or 1).
    """
    # Apply binary cross-entropy loss between predictions and true values
    return nn.BCELoss()(y_pred, y_true)


class LFR(Preprocessor[DataFrame]):
    """
    Learning Fair Representations (LFR) model.

    This preprocessing algorithm learns a fair representation of the data that preserves
    predictive information while removing sensitive attribute information. It uses an
    encoder-decoder architecture with fairness constraints to achieve statistical parity.

    The model optimizes three objectives:
    1. Fairness loss (Lz): Minimizes differences between protected and unprotected groups
    2. Reconstruction loss (Lx): Preserves information from original features
    3. Classification loss (Ly): Maintains predictive power for the target variable
    """

    def __init__(
        self,
        input_dim: Optional[int] = None,
        latent_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        encoder: Optional[Encoder] = None,
        decoder: Optional[Decoder] = None,
        classifier: Optional[Classifier] = None,
        alpha_z: float = 1.0,
        alpha_x: float = 1.0,
        alpha_y: float = 1.0,
        *args,
        **kwargs,
    ):
        """
        Initialize the Learning Fair Representations (LFR) model.

        Parameters
        ----------
        input_dim : Optional[int], default=None
            Dimension of input features
        latent_dim : Optional[int], default=None
            Dimension of the learned fair representation
        output_dim : Optional[int], default=None
            Dimension of the output features (original feature space)
        encoder : Optional[Encoder], default=None
            Custom encoder network (if None, a default encoder will be created)
        decoder : Optional[Decoder], default=None
            Custom decoder network (if None, a default decoder will be created)
        classifier : Optional[Classifier], default=None
            Custom classifier network (if None, a default classifier will be created)
        alpha_z : float, default=1.0
            Weight for the fairness loss (Lz)
        alpha_x : float, default=1.0
            Weight for the reconstruction loss (Lx)
        alpha_y : float, default=1.0
            Weight for the classification loss (Ly)
        """
        # Initialize networks: either use provided custom networks or create default ones
        super().__init__(*args, **kwargs)
        if encoder is None and decoder is None and classifier is None:
            if input_dim is None or latent_dim is None or output_dim is None:
                raise ValueError(
                    "input_dim, latent_dim, and output_dim must be provided when not using custom networks"
                )
            self.encoder = Encoder(input_dim, latent_dim)
            self.decoder = Decoder(latent_dim, output_dim)
            self.classifier = Classifier(latent_dim)
        elif encoder is None or decoder is None or classifier is None:
            raise ValueError(
                "All three networks (encoder, decoder, classifier) must be provided or none"
            )
        else:
            self.encoder = encoder
            self.decoder = decoder
            self.classifier = classifier

        self.alpha_z = alpha_z
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

        # Initialize scaler
        self.scaler = StandardScaler()

    def fit(self, X: DataFrame, y: Optional[np.ndarray] = None, **kwargs) -> "LFR":
        """
        Fit the LFR model to the data.

        Parameters
        ----------
        X : DataFrame
            Input data with numeric features and metadata on target and sensitive columns.
            Must contain exactly one sensitive attribute column.
        y : Optional[np.ndarray], default=None
            Optional target values (if not included in X)
        **kwargs : dict
            Additional parameters including:
            - epochs : int, default=100
                Number of training epochs
            - learning_rate : float, default=0.001
                Learning rate for the optimizer
            - batch_size : int, default=None
                Batch size for training (None means full batch)

        Returns
        -------
        LFR
            The fitted model instance (self)

        Raises
        ------
        ValueError
            If the input DataFrame does not contain exactly one sensitive attribute
        """
        # Extract hyperparameters from kwargs with defaults
        epochs = kwargs.get("epochs", 100)
        learning_rate = kwargs.get("learning_rate", 0.001)

        # Validate that input data has exactly one sensitive attribute
        validate_dataframe(X, expected_sensitive_count=1)

        # Extract features, targets, and sensitive attributes from the DataFrame
        if y is None:
            # If y is not provided, extract it from the DataFrame
            features, targets, _, _, _, sensitive_indexes = X.unpack()
        else:
            # If y is provided separately, use it directly
            features, _, _, _, _, sensitive_indexes = X.unpack()
            targets = y

        # Extract the sensitive attribute values (assumes binary 0/1 encoding)
        sensitive_values = features[:, sensitive_indexes[0]]

        # Fit the model using the extracted data
        self._fit(
            features,
            targets,
            sensitive_values,
            epochs=epochs,
            learning_rate=learning_rate,
        )

        return self

    def fit_transform(
        self, X: DataFrame, y: Optional[np.ndarray] = None, **kwargs
    ) -> DataFrame:
        """
        Fit the LFR model to the data and transform it into fair representations.

        Parameters
        ----------
        X : DataFrame
            Input data with numeric features and metadata on target and sensitive columns.
            Must contain exactly one sensitive attribute column.
        y : Optional[np.ndarray], default=None
            Optional target values (if not included in X)
        **kwargs : dict
            Additional parameters including:
            - epochs : int, default=100
                Number of training epochs
            - learning_rate : float, default=0.001
                Learning rate for the optimizer
            - batch_size : int, default=None
                Batch size for training (None means full batch)

        Returns
        -------
        DataFrame
            Transformed data with fair representations that maintain the same
            metadata structure (sensitive attributes and targets) as the input

        Raises
        ------
        ValueError
            If the input DataFrame does not contain exactly one sensitive attribute
        """
        # First fit the model
        self.fit(X, y, **kwargs)

        # Then transform the data
        return self.transform(X)

    def _fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sensitive_values: np.ndarray,
        epochs: int,
        learning_rate: float,
    ) -> None:
        """
        Fit the LFR model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input features to fit
        y : np.ndarray
            Target values
        sensitive_values : np.ndarray
            Sensitive attribute values (binary: 0 for unprotected, 1 for protected)
        epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate for the optimizer

        Notes
        -----
        This internal method handles the actual training process, including:
        - Data preprocessing (standardization)
        - Converting numpy arrays to PyTorch tensors
        - Setting up the optimizer
        - Training loop with forward/backward passes
        - Loss computation and optimization
        """
        # Standardize features and convert to PyTorch tensors
        X_scaled = self.scaler.fit_transform(X)
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(y).reshape(-1, 1)
        sensitive_tensor = torch.FloatTensor(sensitive_values)

        # Create optimizer for all network parameters
        optimizer = optim.Adam(
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.classifier.parameters()),
            lr=learning_rate,
        )

        # Training loop
        for epoch in range(epochs):
            # Forward pass through all networks
            z = self.encoder(X_tensor)
            x_reconstructed = self.decoder(z)
            y_pred = self.classifier(z)

            # Compute the three loss components
            fairness_loss = compute_fairness_loss(z, sensitive_tensor)
            reconstruction_loss = compute_reconstruction_loss(X_tensor, x_reconstructed)
            classification_loss = compute_classification_loss(y_pred, y_tensor)

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

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Transform input data into fair representations using the trained LFR model.

        Parameters
        ----------
        X : DataFrame
            Input data with numeric features and metadata on target and sensitive columns

        Returns
        -------
        DataFrame
            Fair representations of the input data with reduced disparate impact
        """
        # Validate the input DataFrame
        validate_dataframe(X, expected_sensitive_count=1)

        # Extract features from the DataFrame
        features, _, _, _, _, _ = X.unpack()

        # Transform the features using the internal method
        fair_representations = self._transform(features)

        # Create column names for the latent dimensions
        latent_dim = fair_representations.shape[1]
        latent_cols = [f"z{i}" for i in range(latent_dim)]

        # Create a new DataFrame with the transformed data
        result = DataFrame(fair_representations, columns=latent_cols)

        # We don't set targets or sensitive attributes on the transformed data
        # as these columns don't exist in the latent representation

        return result

    def predict(self, X: DataFrame) -> np.ndarray:
        """
        Predict target values using the trained classifier on fair representations.

        Parameters
        ----------
        X : DataFrame
            Input data with numeric features and metadata on target and sensitive columns

        Returns
        -------
        np.ndarray
            Predicted target values (binary classification probabilities)

        Notes
        -----
        This method first transforms the input data into fair representations using
        the encoder, then applies the classifier to predict target values.
        """
        # Validate the input DataFrame
        validate_dataframe(X, expected_sensitive_count=1)

        # Extract features from the DataFrame
        features, _, _, _, _, _ = X.unpack()

        # Standardize the input features
        X_scaled = self.scaler.transform(features)

        # Convert to tensor
        X_tensor = torch.FloatTensor(X_scaled)

        # Get fair representations and predictions
        with torch.no_grad():
            z = self.encoder(X_tensor)
            y_pred = self.classifier(z)

        return y_pred.numpy()

    def score(self, X: DataFrame, y: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Evaluate the model's performance and fairness metrics.

        Parameters
        ----------
        X : DataFrame
            Input data with numeric features and metadata on target and sensitive columns
        y : Optional[np.ndarray], default=None
            Optional target values (if not included in X)

        Returns
        -------
        Dict[str, float]
            Dictionary containing performance and fairness metrics:
            - 'accuracy': Classification accuracy
            - 'fairness_loss': Statistical parity loss value
            - 'reconstruction_loss': Reconstruction loss value

        Notes
        -----
        This method evaluates both the utility (accuracy) and fairness of the model
        on the provided data, which is useful for assessing the fairness-utility tradeoff.
        """
        # Validate the input DataFrame
        validate_dataframe(X, expected_sensitive_count=1)

        # Extract features, targets, and sensitive attributes
        if y is None:
            features, targets, _, _, _, sensitive_indexes = X.unpack()
        else:
            features, _, _, _, _, sensitive_indexes = X.unpack()
            targets = y

        sensitive_values = features[:, sensitive_indexes[0]]

        # Standardize features and convert to tensors
        X_scaled = self.scaler.transform(features)
        X_tensor = torch.FloatTensor(X_scaled)
        y_tensor = torch.FloatTensor(targets).reshape(-1, 1)
        sensitive_tensor = torch.FloatTensor(sensitive_values)

        # Forward pass through the model
        with torch.no_grad():
            # Get fair representations
            z = self.encoder(X_tensor)

            # Get reconstructed features
            x_reconstructed = self.decoder(z)

            # Get predictions
            y_pred = self.classifier(z)

            # Calculate losses
            fairness_loss = compute_fairness_loss(z, sensitive_tensor).item()
            reconstruction_loss = compute_reconstruction_loss(
                X_tensor, x_reconstructed
            ).item()

            # Calculate accuracy
            predictions = (y_pred > 0.5).float()
            correct = (predictions == y_tensor).sum().item()
            accuracy = correct / len(y_tensor)

        # Return metrics as a dictionary
        return {
            "accuracy": accuracy,
            "fairness_loss": fairness_loss,
            "reconstruction_loss": reconstruction_loss,
        }

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform the input data into fair representations using the trained LFR model.

        Parameters
        ----------
        X : np.ndarray
            Input features to transform

        Returns
        -------
        np.ndarray
            Fair representations of the input data with reduced disparate impact

        Notes
        -----
        This method applies the trained encoder to new data to generate fair representations.
        The transformation process includes:
        1. Standardizing features using the fitted scaler
        2. Converting to PyTorch tensors
        3. Passing through the encoder network
        4. Converting back to numpy arrays
        """
        # Standardize the input features using the fitted scaler
        X_scaled = self.scaler.transform(X)

        # Convert to tensor and get fair representations
        X_tensor = torch.FloatTensor(X_scaled)
        with torch.no_grad():
            z = self.encoder(X_tensor)

        return z.numpy()

    def save(self, path: str) -> None:
        """
        Save the trained LFR model to disk.

        Parameters
        ----------
        path : str
            Path where the model will be saved

        Notes
        -----
        This method saves all components of the LFR model, including:
        - Encoder, decoder, and classifier networks
        - Scaler for feature standardization
        - Loss weights (alpha parameters)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Prepare model components for saving
        model_data = {
            "encoder_state_dict": self.encoder.state_dict(),
            "decoder_state_dict": self.decoder.state_dict(),
            "classifier_state_dict": self.classifier.state_dict(),
            "scaler": self.scaler,
            "alpha_z": self.alpha_z,
            "alpha_x": self.alpha_x,
            "alpha_y": self.alpha_y,
        }

        # Save to disk
        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        input_dim: Optional[int] = None,
        latent_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
    ) -> "LFR":
        """
        Load a trained LFR model from disk.

        Parameters
        ----------
        path : str
            Path to the saved model file
        input_dim : Optional[int], default=None
            Dimension of input features (required if not inferrable from saved model)
        latent_dim : Optional[int], default=None
            Dimension of the latent representation (required if not inferrable from saved model)
        output_dim : Optional[int], default=None
            Dimension of output features (required if not inferrable from saved model)

        Returns
        -------
        LFR
            Loaded LFR model instance

        Raises
        ------
        FileNotFoundError
            If the model file does not exist
        ValueError
            If dimensions cannot be inferred and are not provided
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        # Load model data
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        # Create empty model
        model = cls(input_dim=input_dim, latent_dim=latent_dim, output_dim=output_dim)

        # Load state dictionaries
        model.encoder.load_state_dict(model_data["encoder_state_dict"])
        model.decoder.load_state_dict(model_data["decoder_state_dict"])
        model.classifier.load_state_dict(model_data["classifier_state_dict"])

        # Load other components
        model.scaler = model_data["scaler"]
        model.alpha_z = model_data["alpha_z"]
        model.alpha_x = model_data["alpha_x"]
        model.alpha_y = model_data["alpha_y"]

        # Set model to evaluation mode
        model.encoder.eval()
        model.decoder.eval()
        model.classifier.eval()

        logger.info(f"Model loaded from {path}")
        return model
