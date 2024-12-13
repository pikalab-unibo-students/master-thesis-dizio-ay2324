from fairlib.keras import Model
from fairlib.keras import layers


class Encoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.encoder = [
            layers.Dense(64, input_dim=input_dim),
            layers.ReLU(),
            layers.Dense(32),
            layers.ReLU(),
            layers.Dense(latent_dim),
        ]

    def call(self, x):
        for layer in self.encoder:
            x = layer(x)
        return x


class Decoder(Model):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.decoder = [
            layers.Dense(32, input_dim=latent_dim),
            layers.ReLU(),
            layers.Dense(64),
            layers.ReLU(),
            layers.Dense(output_dim),
        ]

    def call(self, z):
        for layer in self.decoder:
            z = layer(z)
        return z


class Classifier(Model):
    def __init__(self, latent_dim):
        super(Classifier, self).__init__()
        self.classifier = [
            layers.Dense(32, input_dim=latent_dim),
            layers.ReLU(),
            layers.Dense(1, activation='sigmoid'),
        ]

    def call(self, z):
        for layer in self.classifier:
            z = layer(z)
        return z
