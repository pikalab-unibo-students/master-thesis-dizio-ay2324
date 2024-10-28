from fairlib.keras import backend as K
from fairlib.keras import ops as ops
from typing import Optional


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
    y_pred = ops.clip(y_pred, K.epsilon(), 1 - K.epsilon())

    #  Groups based on the sensitive attribute
    group_positive = ops.cast(ops.equal(sensitive_attr, 1), K.floatx())
    group_negative = ops.cast(ops.equal(sensitive_attr, 0), K.floatx())

    # Average probability of a positive result for each group
    prob_positive = ops.sum(y_pred * group_positive) / (ops.sum(group_positive) + K.epsilon())
    prob_negative = ops.sum(y_pred * group_negative) / (ops.sum(group_negative) + K.epsilon())

    # Statistical Parity Difference
    spd = ops.abs(prob_positive - prob_negative)

    return spd


def disparate_impact(y_true, y_pred, sensitive_attr):
    # TODO: IMPLEMENT IT
    return 0


def get(name: str) -> Optional[callable]:
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
        return disparate_impact
    return None
