import torch
from typing import Optional, Callable


def _get_probability_for_spd_and_di(y_pred, sensitive_attr):
    group_positive = (sensitive_attr == 1).float()
    group_negative = (sensitive_attr == 0).float()

    prob_positive = torch.sum(y_pred * group_positive) / (
        torch.sum(group_positive) + 1e-7
    )
    prob_negative = torch.sum(y_pred * group_negative) / (
        torch.sum(group_negative) + 1e-7
    )

    return prob_positive, prob_negative


def _statistical_parity_difference_loss(y_true, y_pred, sensitive_attr_index):
    prob_positive, prob_negative = _get_probability_for_spd_and_di(
        y_pred, sensitive_attr_index
    )

    spd = torch.abs(prob_positive - prob_negative)
    return spd


def _disparate_impact_loss(y_true, y_pred, sensitive_attr_index):
    prob_positive, prob_negative = _get_probability_for_spd_and_di(
        y_pred, sensitive_attr_index
    )

    di = prob_negative / (prob_positive + 1e-7)
    return di


def get(name: str) -> Optional[Callable]:
    if name.lower() in {
        "statistical_parity_difference",
        "statistical_parity",
        "spd",
        "sp",
    }:
        return _statistical_parity_difference_loss
    elif name.lower() in {"disparate_impact", "di"}:
        return _disparate_impact_loss
    return None
