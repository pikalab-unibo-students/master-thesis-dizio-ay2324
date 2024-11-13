from fairlib.keras import backend as keras_backend
from fairlib.keras import ops as keras_ops
from typing import Optional, Callable


def _statistical_parity_difference_loss(y_true, y_pred, sensitive_attr):
    """
    Loss function calculating statistical parity difference.

    Arguments:
    y_true -- tensor of observed values (label)
    y_pred -- tensors of the predictions made by the model
    sensitive_attr -- sensitive attribute (0 or 1) defining groups

    Returns:
    A measure of statistical parity difference as a loss.
    """
    # Ensure that y_pred is limited between 0 and 1
    # epsilon: used to prevent division by 0
    y_pred = keras_ops.clip(
        y_pred, keras_backend.epsilon(), 1 - keras_backend.epsilon()
    )

    #  Groups based on the sensitive attribute
    group_positive = keras_ops.cast(
        keras_ops.equal(sensitive_attr, 1), keras_backend.floatx()
    )
    group_negative = keras_ops.cast(
        keras_ops.equal(sensitive_attr, 0), keras_backend.floatx()
    )

    # Average probability of a positive result for each group
    prob_positive = keras_ops.sum(y_pred * group_positive) / (
        keras_ops.sum(group_positive) + keras_backend.epsilon()
    )
    prob_negative = keras_ops.sum(y_pred * group_negative) / (
        keras_ops.sum(group_negative) + keras_backend.epsilon()
    )

    # Statistical Parity Difference
    spd = keras_ops.abs(prob_positive - prob_negative)

    return spd


def _disparate_impact_loss(y_true, y_pred, sensitive_attr):
    """
    Loss function calculating disparate impact.

    Arguments:
    y_true -- tensor of observed values (label)
    y_pred -- tensors of the predictions made by the model
    sensitive_attr -- sensitive attribute (0 or 1) defining groups

    Returns:
    A measure of disparate impact as a loss.
    """
    # Ensure that y_pred is limited between 0 and 1
    y_pred = keras_ops.clip(
        y_pred, keras_backend.epsilon(), 1 - keras_backend.epsilon()
    )

    # Groups based on the sensitive attribute
    group_positive = keras_ops.cast(
        keras_ops.equal(sensitive_attr, 1), keras_backend.floatx()
    )
    group_negative = keras_ops.cast(
        keras_ops.equal(sensitive_attr, 0), keras_backend.floatx()
    )

    # Average probability of a positive result for each group
    prob_positive = keras_ops.sum(y_pred * group_positive) / (
        keras_ops.sum(group_positive) + keras_backend.epsilon()
    )
    prob_negative = keras_ops.sum(y_pred * group_negative) / (
        keras_ops.sum(group_negative) + keras_backend.epsilon()
    )

    # Disparate Impact
    di = prob_negative / (prob_positive + keras_backend.epsilon())

    return di


def get(name: str) -> Optional[Callable]:
    """
    Returns a metric by name.

    Args:
        name: The name of the metric.

    Returns:
        Metric: The metric instance or None if the metric is not found.
    """
    if name.lower() in {"statistical_parity_difference", "statistical_parity", "sp"}:
        return _statistical_parity_difference_loss
    elif name.lower() in {"disparate_impact", "di"}:
        return _disparate_impact_loss
    return None
