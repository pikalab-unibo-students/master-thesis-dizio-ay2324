from pandas.api.types import is_numeric_dtype
import warnings
import numpy as np


def _get_values(values: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.unique(values)
    values.sort()
    length = len(values)
    if len(values) == 2:
        length -= 1
        if values[0] == 0:
            values = values[::-1]
    return values, length


def check_and_setup(as_dict, sensitive_column, target_column):
    if any(not is_numeric_dtype(c) for c in [target_column, sensitive_column]):
        raise ValueError("Target and sensitive columns must be numeric")
    if len(target_column) != len(sensitive_column):
        raise ValueError("Target and sensitive columns must have the same length")
    target_values, target_len = _get_values(target_column)
    sensitive_values, sensitive_len = _get_values(sensitive_column)
    if sensitive_len < 2:
        warnings.warn(
            f"Sensitive column has less than 2 unique values: {sensitive_len}"
        )
    result = np.zeros((target_len, sensitive_len)) if not as_dict else {}
    return result, sensitive_len, sensitive_values, target_len, target_values
