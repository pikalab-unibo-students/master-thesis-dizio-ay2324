"""Utility functions and common operations for preprocessing algorithms."""

import numpy as np
import pandas as pd
from typing import Callable, Dict, List, Tuple, Union, Optional, Any


def validate_dataframe(df, expected_sensitive_count: Optional[int] = None) -> None:
    """
    Validate that the input is a proper DataFrame with expected structure.

    Parameters
    ----------
    df : DataFrame
        Input data to validate
    expected_sensitive_count : int, optional
        If provided, validates that the DataFrame has exactly this many sensitive attributes

    Raises
    ------
    TypeError
        If df is not a fairlib DataFrame
    ValueError
        If sensitive attribute count doesn't match expected_sensitive_count
    """
    from fairlib import DataFrame

    if not isinstance(df, DataFrame):
        raise TypeError(f"Expected a fairlib DataFrame, got {type(df)}")

    if expected_sensitive_count is not None:
        if len(df.sensitive) != expected_sensitive_count:
            raise ValueError(
                f"Expected {expected_sensitive_count} sensitive column(s), "
                f"got {len(df.sensitive)}: {df.sensitive}"
            )


def validate_target_count(df, expected_count: int = 1) -> None:
    """
    Validate that the DataFrame has the expected number of target columns.

    Parameters
    ----------
    df : DataFrame
        Input data to validate
    expected_count : int, default=1
        Expected number of target columns

    Raises
    ------
    ValueError
        If target column count doesn't match expected_count
    """
    if len(df.targets) != expected_count:
        raise ValueError(
            f"Expected {expected_count} target column(s), "
            f"got {len(df.targets)}: {df.targets}"
        )


def get_privileged_unprivileged_masks(
    df, sensitive_columns: Optional[List[str]] = None
) -> Tuple[pd.Series, pd.Series]:
    """
    Create boolean masks for privileged and unprivileged groups based on sensitive attributes.

    Parameters
    ----------
    df : DataFrame
        Input data with sensitive attributes
    sensitive_columns : List[str], optional
        List of sensitive column names to use. If None, uses all df.sensitive columns.

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (privileged_mask, unprivileged_mask) as boolean Series
    """
    if sensitive_columns is None:
        sensitive_columns = df.sensitive

    privileged = pd.Series([True] * len(df))
    unprivileged = pd.Series([True] * len(df))

    for sensitive_column in sensitive_columns:
        privileged &= df[sensitive_column] == 1
        unprivileged &= df[sensitive_column] == 0

    return privileged, unprivileged


def get_favorable_unfavorable_masks(
    df, target_column: str, favorable_label: int = 1
) -> Tuple[pd.Series, pd.Series]:
    """
    Create boolean masks for favorable and unfavorable outcomes based on target column.

    Parameters
    ----------
    df : DataFrame
        Input data with target column
    target_column : str
        Name of the target column
    favorable_label : int, default=1
        The label value considered favorable

    Returns
    -------
    Tuple[pd.Series, pd.Series]
        (favorable_mask, unfavorable_mask) as boolean Series
    """
    favorable = df[target_column] == favorable_label
    unfavorable = df[target_column] != favorable_label

    return favorable, unfavorable
