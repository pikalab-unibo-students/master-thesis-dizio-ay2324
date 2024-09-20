from fairlib import DataFrameExtensionFunction
from pandas.api.types import is_numeric_dtype
import pandas as pd


class Metric:
    def __call__(self, df):
        raise NotImplementedError

    def apply(self, name):
        DataFrameExtensionFunction(callable=self).apply(name=name)


class StatisticalParityDifference(Metric):
    def __call__(self, df):

        spd = {}

        for target_column in df.targets:
            spd[target_column] = {}
            for group_column in df.sensitive:
                if not is_numeric_dtype(df[target_column]) or not is_numeric_dtype(
                    df[group_column]
                ):
                    raise ValueError(
                        "Target and group columns must be numeric"
                    )

                privileged_group = df[df[group_column] == 1]
                privileged_positive_rate = privileged_group[target_column].mean()
                
                unprivileged_group = df[df[group_column] == 0]
                unprivileged_positive_rate = unprivileged_group[target_column].mean()

                spd_value = privileged_positive_rate - unprivileged_positive_rate
                spd[target_column][group_column] = spd_value
        return spd


class DisparateImpact(Metric):
    def __call__(self, df):
        di = {}

        for target_column in df.targets:
            di[target_column] = {}
            for group_column in df.sensitive:
                if not is_numeric_dtype(df[target_column]) or not is_numeric_dtype(
                    df[group_column]
                ):
                    raise ValueError(
                        "Target and group columns must be numeric"
                    )

                privileged_group = df[df[group_column] == 1]
                privileged_positive_rate = privileged_group[target_column].mean()

                unprivileged_group = df[df[group_column] == 0]
                unprivileged_positive_rate = unprivileged_group[target_column].mean()

                if privileged_positive_rate == 0:
                    return float("inf")  
                di_value = unprivileged_positive_rate / privileged_positive_rate
                di[target_column][group_column] = di_value
        return di



StatisticalParityDifference().apply("statistical_parity_difference")
DisparateImpact().apply("disparate_impact")


