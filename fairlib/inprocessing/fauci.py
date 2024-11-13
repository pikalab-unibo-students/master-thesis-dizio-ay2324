from typing_extensions import override

from ._keras_metrics import get as get_metric
from fairlib.processing import *
import fairlib.keras as keras
from typing import Optional, Any
from keras import regularizers


class RegularizedLoss(keras.losses.Loss):
    """
    A custom loss class that adds regularization to the loss function.

    Attributes:
        __loss (keras.losses.Loss): The base loss function.
        __model (keras.Model): The Keras model.
        __regularization_weight (float): The weight of the regularization term.
    """

    def __init__(
        self,
        loss: Union[str, keras.losses.Loss],
        model: keras.Model,
        regularization_weight: float,
    ):
        """
        Initializes the RegularizedLoss class.

        Args:
            loss (Union[str, keras.losses.Loss]): The base loss function or its name.
            model (keras.Model): The Keras model.
            regularization_weight (float): The weight of the regularization term.
        """
        super().__init__()
        self.__loss: keras.losses.Loss = (
            keras.losses.get(loss) if isinstance(loss, str) else loss
        )
        self.__model: keras.Model = model
        self.__regularization_weight: float = regularization_weight

    def call(self, y_true, y_pred):
        """
        Computes the loss with regularization.

        Args:
            y_true: Ground truth values.
            y_pred: Predicted values.

        Returns:
            The computed loss with regularization.
        """
        loss = self.loss(self.__loss, y_true, y_pred)
        regularization = self.regularization(self.__model, y_true, y_pred)
        weight = keras.ops.convert_to_tensor(self.__regularization_weight)
        return (keras.ops.convert_to_tensor(1.0 - weight) * loss) + (
            weight * keras.ops.convert_to_tensor(regularization)
        )

    def loss(self, f, y_true, y_pred):
        """
        Computes the base loss.

        Args:
            f: The base loss function.
            y_true: Ground truth values.
            y_pred: Predicted values.

        Returns:
            The computed base loss.
        """
        return f(y_true, y_pred)

    def regularization(self, model: keras.Model, y_true, y_pred):
        """
        Computes the regularization term.

        Args:
            model (keras.Model): The Keras model.
            y_true: Ground truth values.
            y_pred: Predicted values.

        Returns:
            The computed regularization term.
        """
        return 0.0


class PenalizedLoss(RegularizedLoss):
    """
    A custom loss class that adds a penalty term based on a specified fairness metric.

    Attributes:
        __metric: The metric used for the penalty term.
        __sensitive_feature: The sensitive feature used for the penalty term.
    """

    def __init__(
        self,
        loss: Union[str, keras.losses.Loss],
        model: keras.Model,
        regularization_weight: float,
        metric: str,
    ):
        """
        Initializes the PenalizedLoss class.

        Args:
            loss (Union[str, keras.losses.Loss]): The base loss function or its name.
            model (keras.Model): The Keras model.
            regularization_weight (float): The weight of the regularization term.
            metric (str): The name of the metric used for the penalty term.
        """
        super().__init__(loss, model, regularization_weight)
        self.__metric = get_metric(metric)
        if self.__metric is None:
            raise ValueError(f"Unknown metric: {metric}")
        self.__sensitive_feature = None

    @property
    def metric(self):
        """
        Returns the metric used for the penalty term.

        Returns:
            The metric used for the penalty term.
        """
        return self.__metric

    @property
    def sensitive_feature(self):
        """
        Returns the sensitive feature used for the penalty term.

        Returns:
            The sensitive feature used for the penalty term.
        """
        return self.__sensitive_feature

    @sensitive_feature.setter
    def sensitive_feature(self, value):
        """
        Sets the sensitive feature used for the penalty term.

        Args:
            value: The sensitive feature.
        """
        self.__sensitive_feature = value

    def regularization(self, model: keras.Model, y_true, y_pred):
        """
        Computes the penalty term based on the specified metric.

        Args:
            model (keras.Model): The Keras model.
            y_true: Ground truth values.
            y_pred: Predicted values.

        Returns:
            The computed penalty term.
        """
        return self.metric(
            y_true=y_true, y_pred=y_pred, sensitive_attr=self.sensitive_feature
        )


class FaUCI(
    DataFrameAwareProcessorWrapper,
    DataFrameAwareEstimator,
    DataFrameAwarePredictor,
    DataFrameAwareModel,
):
    """
    A custom model wrapper that integrates fairness-aware processing with Keras models.

    Attributes:
        __compilation_parameters: The parameters used for model compilation.
        __loss: The loss function used for training.
    """

    def __init__(
        self,
        model: keras.Model,
        loss: Union[str, keras.losses.Loss],
        regularizer: Optional[str] = None,
        regularization_weight: float = 0.5,
        **kwargs,
    ):
        """
        Initializes the FaUCI class.

        Args:
            model (keras.Model): The Keras model.
            loss (Union[str, keras.losses.Loss]): The base loss function or its name.
            regularizer (Optional[str]): The name of the regularizer metric.
            regularization_weight (float): The weight of the regularization term.
            **kwargs: Additional parameters for model compilation.
        """
        if not isinstance(model, keras.Model):
            raise TypeError(f"Expected a Keras model, got {type(model)}")
        super().__init__(model)
        self.__compilation_parameters = kwargs
        if regularizer is None or regularization_weight == 0.0:
            self.__loss = RegularizedLoss(loss, model, regularization_weight)
        else:
            self.__loss = PenalizedLoss(loss, model, regularization_weight, regularizer)

    @override
    def fit(
        self,
        x: DataFrame,
        y: Optional[Any] = None,
        converting_to_type: Optional[type] = None,
        **kwargs,
    ):
        """
        Fits the model to the data.

        Args:
            x (DataFrame): The input data.
            y (Optional[Any]): The target data.
            converting_to_type (Optional[type]): The type to convert the data to.
            **kwargs: Additional parameters for fitting the model.

        Returns:
            The fitted model.
        """
        if not isinstance(x, DataFrame):
            raise TypeError(f"Expected a DataFrame, got {type(x)}")
        x, y, _, _, _, sensitive_indexes = x.unpack()
        if converting_to_type is not None:
            x = x.astype(converting_to_type)
            y = y.astype(converting_to_type)
        y.astype(float)
        if len(sensitive_indexes) != 1:
            raise ValueError(
                f"FaUCI expects exactly one sensitive column, got {len(sensitive_indexes)}: {x.sensitive}"
            )
        if isinstance(self.__loss, PenalizedLoss):
            self.__loss.sensitive_feature = x[:, sensitive_indexes[0]]
        if y.shape[1] != 1:
            raise ValueError(
                f"FaUCI expects exactly one target column, got {y.shape[1]}"
            )
        model: keras.Model = self.processor
        compilation_params = self.__compilation_parameters.copy()
        compilation_params["loss"] = self.__loss
        model.compile(**compilation_params)
        return self._fit(x, y, **kwargs)
