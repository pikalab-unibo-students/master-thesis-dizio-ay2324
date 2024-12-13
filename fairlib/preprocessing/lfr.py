from sklearn.preprocessing import StandardScaler
import numpy as np
from fairlib.keras import losses
from fairlib.keras import optimizers
from fairlib.preprocessing._lfr_model import Encoder, Decoder, Classifier
from fairlib.dataframe import DataFrame


def compute_reconstruction_loss(x, x_reconstructed):
    return np.mean((x - x_reconstructed) ** 2)


def compute_fairness_loss(z, sensitive_attr):
    protected_mask = sensitive_attr == 1
    unprotected_mask = sensitive_attr == 0

    protected_mean = np.mean(z[protected_mask], axis=0)
    unprotected_mean = np.mean(z[unprotected_mask], axis=0)

    return np.sum((protected_mean - unprotected_mean) ** 2)


def compute_classification_loss(y_pred, y_true):
    return losses.BinaryCrossentropy()(y_true, y_pred)


class LFR:
    def __init__(self, input_dim, latent_dim, alpha_z=1.0, alpha_x=1.0, alpha_y=1.0):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.alpha_z = alpha_z  # fairness_loss
        self.alpha_x = alpha_x  # reconstruction_loss
        self.alpha_y = alpha_y  # classification_loss

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
        y = y.reshape(-1, 1)

        self.encoder.compile(optimizer=optimizers.Adam(learning_rate), loss='mse')
        self.decoder.compile(optimizer=optimizers.Adam(learning_rate), loss='mse')
        self.classifier.compile(optimizer=optimizers.Adam(learning_rate), loss=losses.BinaryCrossentropy())

        for epoch in range(epochs):
            z = self.encoder.predict(X)
            x_reconstructed = self.decoder.predict(z)
            y_pred = self.classifier.predict(z)

            fairness_loss = compute_fairness_loss(z, sensitive_attr)
            reconstruction_loss = compute_reconstruction_loss(X, x_reconstructed)
            classification_loss = compute_classification_loss(y_pred, y)

            total_loss = (
                    self.alpha_z * fairness_loss
                    + self.alpha_x * reconstruction_loss
                    + self.alpha_y * classification_loss
            )

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Loss: {total_loss:.4f}, "
                    f"Fairness: {fairness_loss:.4f}, "
                    f"Reconstruction: {reconstruction_loss:.4f}, "
                    f"Classification: {classification_loss:.4f}"
                )

    def predict(self, df: DataFrame):
        X = df.drop(columns=df.targets).values
        X = self.scaler.transform(X)
        z = self.encoder(X)
        y_pred = self.classifier(z)
        return (y_pred.numpy() > 0.5).astype(int)

    def transform(self, df: DataFrame):
        X = df.drop(columns=df.targets).values
        X = self.scaler.transform(X)
        z = self.encoder(X)
        return z.numpy()


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
