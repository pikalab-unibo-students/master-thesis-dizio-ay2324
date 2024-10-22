from keras.src.optimizers import Adam
from fairlib.inprocessing._fauci_metrics import discrete_disparate_impact
from fairlib.processing import *
import fairlib.keras as keras
from fairlib.keras import losses
from fairlib.keras import ops as kops
import fairlib as fl

epsilon = 1e-5

class Fauci:
    # TODO regularizer: Union[str, keras.losses.Loss] ADD THIS TO PARAMETERS
    def __init__(self, model: keras.Model, protected_attribute: int, lambda_value: float, **kwargs):
        if not isinstance(model, keras.Model):
            raise TypeError(f"Expected a Keras model, got {type(model)}")
        self.model = model
        input_layer = model.layers[0].input

        # this is an example
        def tf_disparate_impact(y_true, y_pred):
            return discrete_disparate_impact(input_layer[:, protected_attribute], y_pred)

        # TODO do other implementation for categorical and continuous function

        fairness_metric_function = tf_disparate_impact

        def custom_loss(y_true, y_pred):
            fair_cost_factor = fairness_metric_function(y_true, y_pred)
            # convert the return implemented in tensorflow with keras
            return kops.cast(losses.binary_crossentropy(y_true, y_pred) + epsilon, "float64") + kops.cast(
                lambda_value * fair_cost_factor, "float64")

        self.model.compile(loss=custom_loss, optimizer=Adam(), metrics=["accuracy"])

    def fit(self, train: fl.DataFrame, valid: fl.DataFrame, epochs: int, batch_size: int):
        if not isinstance(train, DataFrame):
            raise TypeError(f"Expected a DataFrame, got {type(train)}")
        if not isinstance(valid, DataFrame):
            raise TypeError(f"Expected a DataFrame, got {type(valid)}")

        sensitive_names = list(train.sensitive)
        if len(sensitive_names) != 1:
            raise ValueError(
                f"FaUCI expects exactly one sensitive column, got {len(sensitive_names)}: {sensitive_names}")

        train_x, train_y = train.iloc[:, :-1], train.iloc[:, -1]
        valid_x, valid_y = valid.iloc[:, :-1], valid.iloc[:, -1]
        return self.model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size,
                              validation_data=(valid_x, valid_y), verbose=0)

    def predict(self, data: fl.DataFrame):
        if not isinstance(data, DataFrame):
            raise TypeError(f"Expected a DataFrame, got {type(data)}")
        return self.model.predict(data)


if __name__ == "__main__":
    algo = Fauci(model=keras.Sequential(...), regularizer="weighted_statistical_parity", optimizer="adam",
                 loss="binary_crossentropy")
