import tensorflow as tf
from fairlib.keras import ops as kops, KerasBackend
from fairlib.keras import initializers
import fairlib.keras as keras
import numpy as np

EPSILON: float = 1e-9
INFINITY: float = 1e9
DELTA: float = 5e-2

class Strategy:
    """
    The strategy to use when comparing the predicted output distribution with the protected attribute.
    """
    EQUAL = 0
    FREQUENCY = 1
    INVERSE_FREQUENCY = 2


def single_conditional_probability(
        predicted: tf.Tensor, protected: tf.Tensor, value: int, equal: bool = True
) -> tf.Tensor:
    """
    Calculate the estimated conditioned output distribution of a model.
    The protected attribute can be binary or categorical.

    @param predicted: the predicted labels.
    @param protected: the protected attribute.
    @param value: the value of the protected attribute.
    @param equal: if True, filter rows whose protected attribute is equal to value, otherwise filter rows whose protected
    attribute is not equal to value.
    @return: the conditional probability.
    """
    mask = kops.cond(
        kops.convert_to_tensor(equal),
        lambda: keras_boolean_mask(predicted, kops.equal(protected, value)),
        lambda: keras_boolean_mask(predicted, kops.not_equal(protected, value)),
    )
    return kops.cond(
        kops.equal(kops.size(mask), 0),
        lambda: initializers.Constant(0.0),
        lambda: kops.mean(mask),
    )


class RegularizedLossFunction:
    def __init__(self, loss_name: str, model: keras.Model, protected_index: int, regularization_weight: float):
        self.__loss = keras.losses.get(loss_name)
        self.__model = model
        self.__protected_index = protected_index
        self.__regularization_weight = regularization_weight

    def __call__(self, y_true, y_pred):
        return self.__loss(y_true, y_pred) + self.__regularization_weight * self.regularization(self.__model,
                                                                                                self.__protected_index)

    def regularization(self, model, protected_index):
        return 0  # possibly this should be a tensor of zeros?


class TensorFlowMixin:
    import tensorflow as tf


class TensorFlowDisparateImpactRegularizedLoss(RegularizedLossFunction, TensorFlowMixin):
    def regularization(self, model, protected_index, y_pred, y_true):

        """
            Calculate the disparate impact of a model.
            The protected attribute is must be binary or categorical.

            @param protected: the protected attribute.
            @param y_pred: the predicted labels.
            @param weights_strategy: the strategy to compute the weights.
            @return: the disparate impact error.
            """
        unique_protected, _ = self.tf.unique(protected)
        impacts = self.tf.map_fn(
            lambda value: tf.reduce_min(
                [
                    self.tf.math.divide_no_nan(
                        single_conditional_probability(y_pred, protected, value),
                        single_conditional_probability(y_pred, protected, value, equal=False),
                    ),
                    self.tf.math.divide_no_nan(
                        single_conditional_probability(y_pred, protected, value, equal=False),
                        single_conditional_probability(y_pred, protected, value),
                    ),
                ]
            ),
            unique_protected,
        )
        if weights_strategy == Strategy.EQUAL:
            return 1 - (tf.reduce_sum(impacts) / tf.cast(tf.size(unique_protected), tf.float32))
        else:
            numbers_a = tf.map_fn(
                lambda value: tf.reduce_sum(tf.cast(tf.equal(protected, value), tf.float32)),
                unique_protected,
            )
            if weights_strategy == Strategy.FREQUENCY:
                return 1 - (tf.reduce_sum(impacts * numbers_a) / tf.reduce_sum(numbers_a))
            elif weights_strategy == Strategy.INVERSE_FREQUENCY:
                return 1 - (tf.reduce_sum(impacts * (tf.reduce_sum(numbers_a) - numbers_a)) / tf.reduce_sum(
                    numbers_a))
