from fairlib import DataFrameExtensionFunction
from pandas.api.types import is_numeric_dtype
import numpy as np
import warnings
from fairlib.utils import *


__all__ = ["Metric", "statistical_parity_difference"]


class Metric:
    """
    Base class for calculating metrics on a dataset. 
    """

    def __call__(self, df, target_column, group_column):
        """
        Should be implemented in subclasses to apply a metric on a DataFrame.

        Args:
            df: The DataFrame containing the data.
            target_column: The column representing the target variable.
            group_column: The column representing the sensitive attribute.
        """
        raise NotImplementedError

    def apply(self, name):
        """
        Applies the extension function to the DataFrame class, using the provided function name.
        As a result of this calling the method "name" on an instance of DataFrame will result in the __call__ method
        being executed.
        """
        DataFrameExtensionFunction(callable=self).apply(name=name)


def __get_values(values: np.ndarray) -> tuple[np.ndarray, int]:
    values = np.unique(values)
    values.sort()
    length = len(values)
    if len(values) == 2:
        length -= 1
        if values[0] == 0:
            values = values[::-1]
    return values, length


def statistical_parity_difference(
        target_column: np.ndarray, 
        sensitive_column: np.ndarray,
        as_dict: bool = False) -> np.ndarray | dict:

    if any(not is_numeric_dtype(c) for c in [target_column, sensitive_column]):
        raise ValueError("Target and sensitive columns must be numeric")

    if len(target_column) != len(sensitive_column):
        raise ValueError("Target and sensitive columns must have the same length")
    
    target_values, target_len = __get_values(target_column)
    sensitive_values, sensitive_len = __get_values(sensitive_column)

    if target_len < 2:
        warnings.warn(f"Target column has less than 2 unique values: {target_values}")
    if sensitive_len < 2:
        warnings.warn(f"Sensitive column has less than 2 unique values: {sensitive_len}")

    result = np.zeros((target_len, sensitive_len)) if not as_dict else {}

    for i in range(target_len):
        target = target_values[i]
        for j in range(sensitive_len):
            sensitive = sensitive_values[j]

            privileged_sensitive = target_column[sensitive_column == sensitive]
            priveleged_sensitive_with_target = privileged_sensitive[privileged_sensitive == target]
            privileged_rate = priveleged_sensitive_with_target.sum() / len(privileged_sensitive) \
                if len(privileged_sensitive) > 0 else np.inf

            unprivileged_sensitive = target_column[sensitive_column != sensitive]
            unprivileged_sensitive_with_target = unprivileged_sensitive[unprivileged_sensitive == target]
            unprivileged_rate = unprivileged_sensitive_with_target.sum() / len(unprivileged_sensitive) \
                if len(unprivileged_sensitive) > 0 else -np.inf

            spd = privileged_rate - unprivileged_rate

            if as_dict:
                result[(target, sensitive)] = spd
            else:
                result[i, j] = spd

    return result


class StatisticalParityDifference(Metric):
    """
    Calculates the Statistical Parity Difference for given target and group columns.

    Args:
        df: The DataFrame containing the data.
        target_columns: List of columns representing the target variable (default is df.targets).
        group_columns: List of columns representing the sensitive attribute (default is df.sensitive).

    Returns:
        dict: A dictionary where keys are target columns and values are dictionaries 
                with group columns as keys and their respective Statistical Parity Difference values.

    Raises:
        ValueError: If target and group columns are not numeric.
    """
    def __call__(self, df, target_columns=None, group_columns=None) -> dict:

        if target_columns is None:
            target_columns = df.targets
        if group_columns is None:
            group_columns = df.sensitive

        result = {}

        for target_column in target_columns:
            for group_column in group_columns:
                # if not is_numeric_dtype(df[target_column]) or not is_numeric_dtype(
                #     df[group_column]
                # ):
                #     raise ValueError("Target and group columns must be numeric")

                # privileged_group = df[df[group_column] == 1]
                # privileged_positive_rate = privileged_group[target_column].mean()

                # unprivileged_group = df[df[group_column] == 0]
                # unprivileged_positive_rate = unprivileged_group[target_column].mean()

                # spd_value = privileged_positive_rate - unprivileged_positive_rate
                # spd[target_column][group_column] = spd_value
                spd = statistical_parity_difference(df[target_column], df[group_column], as_dict=True)
                for (target, group), value in spd.items():
                    result[(Assignment(target_column, target), Assignment(group_column, group))] = value

        return DomainDict(result)


class DisparateImpact(Metric):
    """
    Calculates the Disparate Impact for given target and group columns.

    Args:
        df: The DataFrame containing the data.
        target_columns: List of columns representing the target variable (default is df.targets).
        group_columns: List of columns representing the sensitive attribute (default is df.sensitive).

    Returns:
        dict: A dictionary where keys are target columns and values are dictionaries 
                with group columns as keys and their respective Disparate Impact values.

    Raises:
        ValueError: If target and group columns are not numeric.
    """
    def __call__(self, df, target_columns=None, group_columns=None):

        if target_columns is None:
            target_columns = df.targets
        if group_columns is None:
            group_columns = df.sensitive

        di = {}

        for target_column in target_columns:
            di[target_column] = {}
            for group_column in group_columns:
                if not is_numeric_dtype(df[target_column]) or not is_numeric_dtype(
                    df[group_column]
                ):
                    raise ValueError("Target and group columns must be numeric")

                privileged_group = df[df[group_column] == 1]
                privileged_positive_rate = privileged_group[target_column].mean()

                unprivileged_group = df[df[group_column] == 0]
                unprivileged_positive_rate = unprivileged_group[target_column].mean()

                if privileged_positive_rate == 0:
                    return float("inf")
                di_value = unprivileged_positive_rate / privileged_positive_rate
                di[target_column][group_column] = di_value
        return di


class EqualOpportunityDifference(Metric):
    def __call__(self, df):
        eod = {}

        for target_column in df.targets:
            eod[target_column] = {}
            for group_column in df.sensitive:
                if not is_numeric_dtype(df[target_column]) or not is_numeric_dtype(
                    df[group_column]
                ):
                    raise ValueError("Target and group columns must be numeric")

                privileged_group = df[df[group_column] == 1]

                unprivileged_group = df[df[group_column] == 0]

                eod_value = ...
                eod[target_column][group_column] = eod_value
        raise NotImplementedError


StatisticalParityDifference().apply("statistical_parity_difference")
DisparateImpact().apply("disparate_impact")
