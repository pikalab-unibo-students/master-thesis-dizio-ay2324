from ._keras_metrics import get as get_metric
from fairlib.processing import *
import fairlib.keras as keras


class RegularizedLoss:
    def __init__(self, loss: str | keras.losses.Loss, model: keras.Model, regularization_weight: float):
        self.__loss: keras.losses.Loss = keras.losses.get(loss) if isinstance(loss, str) else loss
        self.__model: keras.Model = model
        self.__regularization_weight: float = regularization_weight

    def __call__(self, y_true, y_pred):
        loss = self.loss(self.__loss, y_true, y_pred)
        regularization = self.regularization(self.__model, y_true, y_pred)
        weight = self.__regularization_weight
        return loss + keras.ops.convert_to_tensor(weight * regularization)

    def loss(self, f, y_true, y_pred):
        return f(y_true, y_pred)

    def regularization(self, model: keras.Model, y_true, y_pred):
        return 0.0


class PenalizedLoss(RegularizedLoss):
    def __init__(self, loss: str | keras.losses.Loss, model: keras.Model, regularization_weight: float, metric: str):
        super().__init__(loss, model, regularization_weight)
        self.__metric = get_metric(metric)
        if self.__metric is None:
            raise ValueError(f"Unknown metric: {metric}")
        self.__sensitive_feature = None

    @property
    def metric(self):
        return self.__metric

    @property
    def sensitive_feature(self):
        return self.__sensitive_feature

    @sensitive_feature.setter
    def sensitive_feature(self, value):
        self.__sensitive_feature = value

    def regularization(self, model: keras.Model, y_true, y_pred):
        return self.metric(y_true=y_true, y_pred=y_pred, sensitive_attr=self.sensitive_feature)


class FaUCI(DataFrameAwareProcessorWrapper, DataFrameAwareEstimator, DataFrameAwarePredictor, DataFrameAwareModel):
    def __init__(self,
                 model: keras.Model,
                 loss: str | keras.losses.Loss,
                 regularizer: str = None,
                 regularization_weight: float = 1.0,
                 **kwargs):
        if not isinstance(model, keras.Model):
            raise TypeError(f"Expected a Keras model, got {type(model)}")
        super().__init__(model)
        self.__compilation_parameters = kwargs
        if regularizer is None:
            self.__loss = RegularizedLoss(loss, model, regularization_weight)
        else:
            self.__loss = PenalizedLoss(loss, model, regularization_weight, regularizer)

    def fit(self, x: DataFrame, converting_to_type: type = None, **kwargs):
        if not isinstance(x, DataFrame):
            raise TypeError(f"Expected a DataFrame, got {type(x)}")
        x, y, _, _, _, sensitive_indexes = x.unpack()
        if converting_to_type is not None:
            x = x.astype(converting_to_type)
            y = y.astype(converting_to_type)
        y.astype(float)
        if len(sensitive_indexes) != 1:
            raise ValueError(f"FaUCI expects exactly one sensitive column, got {len(sensitive_indexes)}: {x.sensitive}")
        if isinstance(self.__loss, PenalizedLoss):
            self.__loss.sensitive_feature = x[:, sensitive_indexes[0]]
        if y.shape[1] != 1:
            raise ValueError(f"FaUCI expects exactly one target column, got {y.shape[1]}")
        model: keras.Model = self.processor
        compilation_params = self.__compilation_parameters.copy()
        compilation_params["loss"] = self.__loss
        model.compile(**compilation_params)
        return self._fit(x, y, **kwargs)

# See notebook for example
