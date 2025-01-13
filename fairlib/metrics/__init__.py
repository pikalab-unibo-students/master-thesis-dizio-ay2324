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
                len(privileged_sensitive_with_target) / len(privileged_sensitive)
                if len(privileged_sensitive) > 0
                else np.inf
            )

            unprivileged_sensitive = target_column[sensitive_column != sensitive]
            unprivileged_sensitive_with_target = unprivileged_sensitive[
                unprivileged_sensitive == target
            ]
            unprivileged_rate = (
                len(unprivileged_sensitive_with_target) / len(unprivileged_sensitive)
                if len(unprivileged_sensitive) > 0
                else -np.inf
            )

            spd = privileged_rate - unprivileged_rate

            if as_dict:
                result[(target, sensitive)] = spd
            else:
                result[j, i] = spd

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
                len(privileged_sensitive_with_target) / len(privileged_sensitive)
                if len(privileged_sensitive) > 0
                else np.inf
            )

            unprivileged_sensitive = target_column[sensitive_column != sensitive]
            unprivileged_sensitive_with_target = unprivileged_sensitive[
                unprivileged_sensitive == target
            ]
            unprivileged_rate = (
                len(unprivileged_sensitive_with_target) / len(unprivileged_sensitive)
                if len(unprivileged_sensitive) > 0
                else -np.inf
            )

            di = unprivileged_rate / privileged_rate

            if as_dict:
                result[(target, sensitive)] = di
            else:
                result[j, i] = di

    return result


def equality_of_opportunity(
    target_column: np.ndarray,
    sensitive_column: np.ndarray,
    predicted_column: np.ndarray,
    positive_target: int = 1,
    as_dict: bool = False,
) -> Union[np.ndarray, dict]:
    """
    Calculates Equality of Opportunity for sensitive groups.

    Args:
        target_column: Array containing the true value of the target.
        sensitive_column: Array containing the value of the sensitive group.
        predicted_column: Array containing the predictions of the model.
        positive_target: Value considered as positive class (default 1).
        as_dict: If True, returns a dictionary with results for each combination.

    Returns:
        np.ndarray or dict: The difference in True Positive Rates between privileged and unprivileged groups.
    """
    result, sensitive_len, sensitive_values, _, _ = check_and_setup(
        as_dict, sensitive_column, target_column
    )

    for j in range(sensitive_len):
        sensitive = sensitive_values[j]

        # TPR for the privileged group
        privileged_mask = sensitive_column == sensitive
        privileged_positive = (target_column == positive_target) & privileged_mask
        privileged_pred_positive = (
            predicted_column == positive_target
        ) & privileged_mask
        privileged_tpr = (
            privileged_pred_positive.sum() / privileged_positive.sum()
            if privileged_positive.sum() > 0
            else np.inf
        )

        # TPR for the non-privileged group
        unprivileged_mask = sensitive_column != sensitive
        unprivileged_positive = (target_column == positive_target) & unprivileged_mask
        unprivileged_pred_positive = (
            predicted_column == positive_target
        ) & unprivileged_mask
        unprivileged_tpr = (
            unprivileged_pred_positive.sum() / unprivileged_positive.sum()
            if unprivileged_positive.sum() > 0
            else -np.inf
        )

        eoo = privileged_tpr - unprivileged_tpr

        if as_dict:
            result[sensitive] = eoo
        else:
            result[j] = eoo

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


class EqualityOfOpportunity(Metric):
    """
    Calculates Equality of Opportunity for target columns, predictions, and sensitive groups.

    Args:
        df: The DataFrame containing the data.
        target_columns: List of columns representing the target (default is df.targets).
        group_columns: List of columns representing the sensitive group (default is df.sensitive).
        prediction_columns: List of columns representing the predictions.

    Returns:
        dict: Dictionary where keys are pairs of target and group columns,
              and the values are the differences in the TPRs.

    Raises:
        ValueError: If the columns are not numeric.

    """

    def __call__(self, df, predictions, target_columns=None, group_columns=None):

        if target_columns is None:
            target_columns = df.targets
        if group_columns is None:
            group_columns = df.sensitive
        if predictions is []:
            raise "predictions cannot be empty"

        result = {}

        for target_column in target_columns:
            for group_column in group_columns:
                eoo = equality_of_opportunity(
                    df[target_column], df[group_column], predictions, as_dict=True
                )
                if isinstance(eoo, dict):
                    for group, value in eoo.items():
                        result[
                            (
                                Assignment(target_column, 1),
                                Assignment(group_column, group),
                            )
                        ] = value
        return DomainDict(result)


StatisticalParityDifference().apply("statistical_parity_difference")
DisparateImpact().apply("disparate_impact")
EqualityOfOpportunity().apply("equality_of_opportunity")
