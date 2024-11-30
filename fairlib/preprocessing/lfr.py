import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from _lfr_model import Encoder, Decoder, Classifier


class LFR:
    def __init__(self, input_dim, latent_dim, alpha_z=1.0, alpha_x=1.0, alpha_y=1.0):
        """
        Learning Fair Representations (LFR) model

        Parameters:
        -----------
        input_dim: int
            Dimension of input features
        latent_dim: int
            Dimension of the learned fair representation
        alpha_z: float
            Weight for the fairness loss (Lz)
        alpha_x: float
            Weight for the reconstruction loss (Lx)
        alpha_y: float
            Weight for the classification loss (Ly)
        """
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.alpha_z = alpha_z
        self.alpha_x = alpha_x
        self.alpha_y = alpha_y

        # Initialize networks
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)
        self.classifier = Classifier(latent_dim)

        # Initialize scaler
        self.scaler = StandardScaler()

    def compute_fairness_loss(self, z, sensitive_attr):
        """
        Compute statistical parity loss (Lz)
        """
        protected_mask = (sensitive_attr == 1)
        unprotected_mask = (sensitive_attr == 0)

        protected_mean = torch.mean(z[protected_mask], dim=0)
        unprotected_mean = torch.mean(z[unprotected_mask], dim=0)

        return torch.sum((protected_mean - unprotected_mean) ** 2)

    def compute_reconstruction_loss(self, x, x_reconstructed):
        """
        Compute reconstruction loss (Lx)
        """
        return torch.mean((x - x_reconstructed) ** 2)

    def compute_classification_loss(self, y_pred, y_true):
        """
        Compute binary cross-entropy loss (Ly)
        """
        return nn.BCELoss()(y_pred, y_true)

    def fit(self, X, y, sensitive_attr, epochs=100, batch_size=32, learning_rate=0.001):
        """
        Train the LFR model
        """
        # Prepare data
        X = self.scaler.fit_transform(X)
        X = torch.FloatTensor(X)
        y = torch.FloatTensor(y).reshape(-1, 1)
        sensitive_attr = torch.FloatTensor(sensitive_attr)

        # Create optimizer
        optimizer = optim.Adam(list(self.encoder.parameters()) +
                               list(self.decoder.parameters()) +
                               list(self.classifier.parameters()),
                               lr=learning_rate)

        # Training loop
        for epoch in range(epochs):
            # Forward pass
            z = self.encoder(X)
            x_reconstructed = self.decoder(z)
            y_pred = self.classifier(z)

            # Compute losses
            fairness_loss = self.compute_fairness_loss(z, sensitive_attr)
            reconstruction_loss = self.compute_reconstruction_loss(X, x_reconstructed)
            classification_loss = self.compute_classification_loss(y_pred, y)

            # Compute total loss
            total_loss = (self.alpha_z * fairness_loss +
                          self.alpha_x * reconstruction_loss +
                          self.alpha_y * classification_loss)

            # Backward pass and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], "
                      f"Loss: {total_loss.item():.4f}, "
                      f"Fairness: {fairness_loss.item():.4f}, "
                      f"Reconstruction: {reconstruction_loss.item():.4f}, "
                      f"Classification: {classification_loss.item():.4f}")

    def predict(self, X):
        """
        Make predictions on new data
        """
        X = self.scaler.transform(X)
        X = torch.FloatTensor(X)
        z = self.encoder(X)
        y_pred = self.classifier(z)
        return (y_pred.detach().numpy() > 0.5).astype(int)

    def transform(self, X):
        """
        Transform data into fair representations
        """
        X = self.scaler.transform(X)
        X = torch.FloatTensor(X)
        z = self.encoder(X)
        return z.detach().numpy()


def main():
    """
    Main function demonstrating LFR with a synthetic dataset containing gender bias
    """
    # Set random seed for reproducibility
    np.random.seed(42)

    # Dataset parameters
    n_samples = 1000
    n_features = 10

    # Generate synthetic features
    X = np.random.randn(n_samples, n_features)

    # Generate gender as sensitive attribute (0: Female, 1: Male)
    # Let's assume 45% females and 55% males
    gender = np.random.binomial(1, 0.55, n_samples)

    # Introduce gender-based disparities:
    # 1. Different feature distributions based on gender
    # 2. Biased outcome generation

    # Add gender-based differences in features
    # Assume feature 0 represents income (with gender pay gap)
    X[:, 0] = X[:, 0] + 0.5 * gender  # Males have higher average income

    # Assume feature 1 represents years of experience
    X[:, 1] = X[:, 1] + 0.3 * gender  # Males have slightly more experience on average

    # Generate target variable (e.g., loan approval)
    # Include both legitimate and discriminatory factors
    legitimate_factors = (
            0.7 * X[:, 0] +  # Income
            0.3 * X[:, 1] +  # Experience
            0.2 * X[:, 2]  # Other legitimate factor
    )

    # Add discrimination: males are favored in the historical data
    discriminatory_bias = 0.5 * gender

    # Add random noise
    noise = 0.1 * np.random.randn(n_samples)

    # Generate final target (loan approval)
    y = (legitimate_factors + discriminatory_bias + noise > 0).astype(int)

    # Split data
    X_train, X_test, y_train, y_test, gender_train, gender_test = train_test_split(
        X, y, gender, test_size=0.2, random_state=42)

    # Initialize and train LFR model
    lfr = LFR(input_dim=n_features,
              latent_dim=8,
              alpha_z=1.0,  # Fairness weight
              alpha_x=0.5,  # Reconstruction weight
              alpha_y=1.0)  # Classification weight

    # Train the model
    lfr.fit(X_train, y_train, gender_train, epochs=100)

    # Make predictions
    y_pred = lfr.predict(X_test)

    # Evaluate overall results
    print("\nOverall Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Evaluate fairness metrics
    def compute_group_metrics(y_true, y_pred, gender):
        female_mask = (gender == 0)
        male_mask = (gender == 1)

        female_acceptance = np.mean(y_pred[female_mask])
        male_acceptance = np.mean(y_pred[male_mask])

        female_accuracy = accuracy_score(y_true[female_mask], y_pred[female_mask])
        male_accuracy = accuracy_score(y_true[male_mask], y_pred[male_mask])

        return female_acceptance, male_acceptance, female_accuracy, male_accuracy

    # Compute metrics for original features
    print("\nFairness Analysis:")
    print("Original Data Demographics:")
    f_acc_orig = np.mean(y_test[gender_test == 0])
    m_acc_orig = np.mean(y_test[gender_test == 1])
    print(f"Female acceptance rate: {f_acc_orig:.3f}")
    print(f"Male acceptance rate: {m_acc_orig:.3f}")
    print(f"Demographic parity difference: {abs(m_acc_orig - f_acc_orig):.3f}")

    # Compute metrics for LFR predictions
    f_acc_lfr, m_acc_lfr, f_accuracy, m_accuracy = compute_group_metrics(
        y_test, y_pred, gender_test)

    print("\nLFR Model Predictions:")
    print(f"Female acceptance rate: {f_acc_lfr:.3f}")
    print(f"Male acceptance rate: {m_acc_lfr:.3f}")
    print(f"Demographic parity difference: {abs(m_acc_lfr - f_acc_lfr):.3f}")
    print(f"Female accuracy: {f_accuracy:.3f}")
    print(f"Male accuracy: {m_accuracy:.3f}")

    # Visualize fair representations
    z_train = lfr.transform(X_train)

    plt.figure(figsize=(15, 5))

    # Plot original features (first two dimensions)
    plt.subplot(1, 2, 1)
    plt.scatter(X_train[gender_train == 0, 0], X_train[gender_train == 0, 1],
                label='Female', alpha=0.5)
    plt.scatter(X_train[gender_train == 1, 0], X_train[gender_train == 1, 1],
                label='Male', alpha=0.5)
    plt.title('Original Feature Space\n(First Two Dimensions)')
    plt.xlabel('Feature 1 (Income)')
    plt.ylabel('Feature 2 (Experience)')
    plt.legend()

    # Plot learned fair representations
    plt.subplot(1, 2, 2)
    plt.scatter(z_train[gender_train == 0, 0], z_train[gender_train == 0, 1],
                label='Female', alpha=0.5)
    plt.scatter(z_train[gender_train == 1, 0], z_train[gender_train == 1, 1],
                label='Male', alpha=0.5)
    plt.title('Learned Fair Representations\n(First Two Dimensions)')
    plt.xlabel('First dimension')
    plt.ylabel('Second dimension')
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
