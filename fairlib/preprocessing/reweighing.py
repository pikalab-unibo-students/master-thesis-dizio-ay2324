import numpy as np
import pandas as pd
from fairlib import DataFrame
from typing import Any, Optional, Tuple

__all__ = ["Reweighing", "ReweighingWithMean"]

from .pre_processing import Preprocessor
from .utils import (
    validate_dataframe,
    validate_target_count,
    get_privileged_unprivileged_masks,
    get_favorable_unfavorable_masks,
)


class Reweighing(Preprocessor[DataFrame]):
    @staticmethod
    def _ratio(
        n_outcome: float, n_group: float, n_total: float, n_joint: float
    ) -> float:
        if n_joint == 0:
            return 0.0
        return (n_outcome * n_group) / (n_total * n_joint)

    @staticmethod
    def _reweighing(
        privileged: pd.Series,
        unprivileged: pd.Series,
        favorable: pd.Series,
        unfavorable: pd.Series,
        n_total: int,
    ) -> Tuple[float, float, float, float]:
        n_favorable = np.sum(favorable)
        n_unfavorable = np.sum(unfavorable)

        n_privileged = np.sum(privileged)
        n_unprivileged = np.sum(unprivileged)

        n_privileged_favorable = np.sum(privileged & favorable)
        n_privileged_unfavorable = np.sum(privileged & unfavorable)
        n_unprivileged_favorable = np.sum(unprivileged & favorable)
        n_unprivileged_unfavorable = np.sum(unprivileged & unfavorable)

        weight_privileged_favorable = Reweighing._ratio(
            n_favorable, n_privileged, n_total, n_privileged_favorable
        )
        weight_privileged_unfavorable = Reweighing._ratio(
            n_unfavorable, n_privileged, n_total, n_privileged_unfavorable
        )
        weight_unprivileged_favorable = Reweighing._ratio(
            n_favorable, n_unprivileged, n_total, n_unprivileged_favorable
        )
        weight_unprivileged_unfavorable = Reweighing._ratio(
            n_unfavorable, n_unprivileged, n_total, n_unprivileged_unfavorable
        )

        return (
            weight_privileged_favorable,
            weight_privileged_unfavorable,
            weight_unprivileged_favorable,
            weight_unprivileged_unfavorable,
        )

    def fit_transform(
        self, X: DataFrame, y: Optional[Any] = None, **kwargs
    ) -> DataFrame:
        """
        Fit the reweighing model and transform the data in one step.

        Parameters
        ----------
        X : DataFrame
            Input data with numeric features and metadata on target and sensitive columns.
        y : ignored, use X.targets and X.sensitive
        favorable_label : int, optional
            The label of the favorable outcome, by default 1

        Returns
        -------
        DataFrame
            Transformed data with reweighting applied.
        """
        favorable_label = kwargs.get("favorable_label", 1)
        return self._transform(X, y, favorable_label)

    def _transform(
        self, df: DataFrame, y: Optional[Any], favorable_label: int
    ) -> DataFrame:
        """
        Apply reweighing transformation to the data.

        Parameters
        ----------
        df : DataFrame
            Input data
        y : Optional[Any]
            Ignored, target is taken from df
        favorable_label : int
            The label value considered favorable

        Returns
        -------
        DataFrame
            Transformed data with weights column added
        """
        # Validate input data
        validate_dataframe(df)
        validate_target_count(df)

        # Get the target column (first one)
        target_column = df.targets.pop()

        # Create masks for favorable/unfavorable outcomes
        favorable, unfavorable = get_favorable_unfavorable_masks(
            df, target_column, favorable_label
        )

        n_total = len(df)
        df["weights"] = 1.0

        # Create masks for privileged/unprivileged groups
        privileged, unprivileged = get_privileged_unprivileged_masks(df)

        # Calculate weights
        (
            weight_privileged_favorable,
            weight_privileged_unfavorable,
            weight_unprivileged_favorable,
            weight_unprivileged_unfavorable,
        ) = self._reweighing(privileged, unprivileged, favorable, unfavorable, n_total)

        # Apply weights
        df.loc[privileged & favorable, "weights"] = weight_privileged_favorable
        df.loc[privileged & unfavorable, "weights"] = weight_privileged_unfavorable
        df.loc[unprivileged & favorable, "weights"] = weight_unprivileged_favorable
        df.loc[unprivileged & unfavorable, "weights"] = weight_unprivileged_unfavorable

        return df


class ReweighingWithMean(Reweighing):
    """
    Enhanced Reweighing that calculates weights for each sensitive attribute separately.

    This variant computes weights for each sensitive attribute independently and then
    takes the mean of all weights as the final instance weight. This can be useful when
    dealing with multiple sensitive attributes.
    """

    def _transform(
        self,
        df: DataFrame,
        y: Optional[Any],
        favorable_label: int,
        remove_weight_columns: bool = True,
    ) -> DataFrame:
        """
        Apply reweighing transformation with per-attribute weights.

        Parameters
        ----------
        df : DataFrame
            Input data
        y : Optional[Any]
            Ignored, target is taken from df
        favorable_label : int
            The label value considered favorable
        remove_weight_columns : bool, default=True
            Whether to remove intermediate weight columns after computing the mean

        Returns
        -------
        DataFrame
            Transformed data with weights column added
        """
        # Validate input data
        validate_dataframe(df)
        validate_target_count(df)

        # Get the target column (first one)
        target_column = df.targets.pop()

        # Create masks for favorable/unfavorable outcomes
        favorable, unfavorable = get_favorable_unfavorable_masks(
            df, target_column, favorable_label
        )

        n_total = len(df)
        df["weights"] = 1.0

        # Calculate weights for each sensitive attribute separately
        weight_columns = []
        for sensitive_column in df.sensitive:
            privileged = df[sensitive_column] == 1
            unprivileged = df[sensitive_column] == 0

            (
                weight_privileged_favorable,
                weight_privileged_unfavorable,
                weight_unprivileged_favorable,
                weight_unprivileged_unfavorable,
            ) = self._reweighing(
                privileged, unprivileged, favorable, unfavorable, n_total
            )

            weight_col_name = f"weights_{sensitive_column}"
            df[weight_col_name] = 1.0

            # Apply weights for this sensitive attribute
            df.loc[privileged & favorable, weight_col_name] = (
                weight_privileged_favorable
            )
            df.loc[privileged & unfavorable, weight_col_name] = (
                weight_privileged_unfavorable
            )
            df.loc[unprivileged & favorable, weight_col_name] = (
                weight_unprivileged_favorable
            )
            df.loc[unprivileged & unfavorable, weight_col_name] = (
                weight_unprivileged_unfavorable
            )

            weight_columns.append(weight_col_name)

        # Calculate mean weight across all sensitive attributes
        df["weights"] = df[weight_columns].mean(axis=1)

        # Optionally remove intermediate weight columns
        if remove_weight_columns:
            df.drop(columns=weight_columns, inplace=True)

        return df
