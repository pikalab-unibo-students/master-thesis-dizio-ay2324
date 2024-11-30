import torch.nn as nn


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
