from fairlib.dataframe import DataFrameExtensionFunction
import numpy as np

from fairlib.metrics._utils import check_and_setup
from fairlib.utils import *

from typing import Union


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


def statistical_parity_difference(
    target_column: np.ndarray, sensitive_column: np.ndarray, as_dict: bool = False
) -> Union[np.ndarray, dict]:

    result, sensitive_len, sensitive_values, target_len, target_values = (
        check_and_setup(as_dict, sensitive_column, target_column)
    )

    for i in range(target_len):
        target = target_values[i]
        for j in range(sensitive_len):
            sensitive = sensitive_values[j]

            privileged_sensitive = target_column[sensitive_column == sensitive]
            privileged_sensitive_with_target = privileged_sensitive[
                privileged_sensitive == target
            ]
            privileged_rate = (
                privileged_sensitive_with_target.sum() / len(privileged_sensitive)
                if len(privileged_sensitive) > 0
                else np.inf
            )

            unprivileged_sensitive = target_column[sensitive_column != sensitive]
            unprivileged_sensitive_with_target = unprivileged_sensitive[
                unprivileged_sensitive == target
            ]
            unprivileged_rate = (
                unprivileged_sensitive_with_target.sum() / len(unprivileged_sensitive)
                if len(unprivileged_sensitive) > 0
                else -np.inf
            )

            spd = privileged_rate - unprivileged_rate

            if as_dict:
                result[(target, sensitive)] = spd
            else:
                result[i, j] = spd

    return result


def disparate_impact(
    target_column: np.ndarray, sensitive_column: np.ndarray, as_dict: bool = False
) -> Union[np.ndarray, dict]:

    result, sensitive_len, sensitive_values, target_len, target_values = (
        check_and_setup(as_dict, sensitive_column, target_column)
    )

    for i in range(target_len):
        target = target_values[i]
        for j in range(sensitive_len):
            sensitive = sensitive_values[j]

            privileged_sensitive = target_column[sensitive_column == sensitive]
            privileged_sensitive_with_target = privileged_sensitive[
                privileged_sensitive == target
            ]
            privileged_rate = (
                privileged_sensitive_with_target.sum() / len(privileged_sensitive)
                if len(privileged_sensitive) > 0
                else np.inf
            )

            unprivileged_sensitive = target_column[sensitive_column != sensitive]
            unprivileged_sensitive_with_target = unprivileged_sensitive[
                unprivileged_sensitive == target
            ]
            unprivileged_rate = (
                unprivileged_sensitive_with_target.sum() / len(unprivileged_sensitive)
                if len(unprivileged_sensitive) > 0
                else -np.inf
            )

            di = unprivileged_rate / privileged_rate

            if as_dict:
                result[(target, sensitive)] = di
            else:
                result[i, j] = di

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

    def __call__(self, df, target_columns=None, group_columns=None) -> DomainDict:

        if target_columns is None:
            target_columns = df.targets
        if group_columns is None:
            group_columns = df.sensitive

        result = {}

        for target_column in target_columns:
            for group_column in group_columns:
                spd = statistical_parity_difference(
                    df[target_column], df[group_column], as_dict=True
                )
                if isinstance(spd, dict):
                    for (target, group), value in spd.items():
                        result[
                            (
                                Assignment(target_column, target),
                                Assignment(group_column, group),
                            )
                        ] = value
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

        result = {}

        for target_column in target_columns:
            for group_column in group_columns:
                di = disparate_impact(df[target_column], df[group_column], as_dict=True)
                if isinstance(di, dict):
                    for (target, group), value in di.items():
                        result[
                            (
                                Assignment(target_column, target),
                                Assignment(group_column, group),
                            )
                        ] = value
        return DomainDict(result)


StatisticalParityDifference().apply("statistical_parity_difference")
DisparateImpact().apply("disparate_impact")
