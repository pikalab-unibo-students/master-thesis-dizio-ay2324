from fairlib import DataFrameExtensionFunction
from pandas.api.types import is_numeric_dtype


class Metric:
    """
    Base class for calculating metrics on a dataset.

    Methods:
    --------
    __call__(df):
        Should be implemented in subclasses to apply a metric on a DataFrame.

    apply(name):
        Applies the extension function on a DataFrame using a function name.
    """
    def __call__(self, df, target_column, group_column):
        raise NotImplementedError

    def apply(self, name):
        DataFrameExtensionFunction(callable=self).apply(name=name)


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
    def __call__(self, df, target_columns=None, group_columns=None):

        if target_columns is None:
            target_columns = df.targets
        if group_columns is None:
            group_columns = df.sensitive

        spd = {}

        for target_column in target_columns:
            spd[target_column] = {}
            for group_column in group_columns:
                if not is_numeric_dtype(df[target_column]) or not is_numeric_dtype(
                    df[group_column]
                ):
                    raise ValueError("Target and group columns must be numeric")

                privileged_group = df[df[group_column] == 1]
                privileged_positive_rate = privileged_group[target_column].mean()

                unprivileged_group = df[df[group_column] == 0]
                unprivileged_positive_rate = unprivileged_group[target_column].mean()

                spd_value = privileged_positive_rate - unprivileged_positive_rate
                spd[target_column][group_column] = spd_value
        return spd


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
