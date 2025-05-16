import numpy as np
from typing import Any, Optional, Dict, List, Callable
from fairlib import DataFrame
from .pre_processing import Preprocessor
from .utils import validate_dataframe


def _make_cdf(values: np.ndarray) -> Callable[[float], float]:
    """
    Create an empirical cumulative distribution function (CDF)
    based on sorted values.
    """
    sorted_vals = np.sort(values)

    def cdf(val: float) -> float:
        # Proportion of samples <= val
        return float(np.searchsorted(sorted_vals, val, side="right") / len(sorted_vals))

    return cdf


class DisparateImpactRemover(Preprocessor[DataFrame]):
    """
    Fairness-aware repair algorithm that removes disparate impact
    by transforming feature distributions to a common (median) distribution
    across sensitive attribute groups, as described in Feldman et al. (2015).

    Parameters
    ----------
    repair_level : float, default=1.0
        Degree of repair:
        - 1.0 = full repair (max fairness, min predictability of sensitive attribute)
        - 0.0 = no repair (original data)
    """

    def __init__(self, repair_level: float = 1.0):
        """
        Initialize the DisparateImpactRemover.

        Parameters
        ----------
        repair_level : float, default=1.0
            Degree of repair:
            - 1.0 = full repair (max fairness, min predictability of sensitive attribute)
            - 0.0 = no repair (original data)
            Values between 0 and 1 provide a trade-off between fairness and utility.
        """
        self.repair_level = repair_level
        self.input_names: Optional[List[str]] = None
        self.target_names: Optional[List[str]] = None
        self.sensitive_names: Optional[List[str]] = None
        self.sensitive_values: Optional[np.ndarray] = None
        self.quantile_maps: Dict[int, Any] = {}

    def fit_transform(
        self, X: DataFrame, y: Optional[Any] = None, **kwargs
    ) -> DataFrame:
        """
        Fit the repair model and transform the data in one step.

        Parameters
        ----------
        X : DataFrame
            Input data with numeric features and metadata on target and sensitive columns.
        y : ignored, use X.targets and X.sensitive
        kwargs : passed to internal _fit

        Returns
        -------
        DataFrame
            Transformed DataFrame with repaired feature values.
        """
        # Validate input data
        validate_dataframe(X, expected_sensitive_count=1)

        # Unpack data
        inputs, _, input_names, target_names, sensitive_names, sensitive_indexes = (
            X.unpack()
        )

        # Store metadata for later use
        self.input_names = input_names
        self.target_names = target_names
        self.sensitive_names = sensitive_names

        # Fit
        self._fit(inputs, sensitive_indexes, **kwargs)
        # Transform
        transformed = self.transform(inputs, sensitive_indexes)

        # Reconstruct DataFrame preserving metadata
        transformed_df = DataFrame(transformed, columns=self.input_names)

        transformed_df.sensitive = self.sensitive_names
        return transformed_df

    def _fit(
        self, inputs: np.ndarray, sensitive_idxs: List[int], **kwargs
    ) -> "DisparateImpactRemover":
        """
        Compute per-group CDFs and median quantile distributions.

        Parameters
        ----------
        inputs : np.ndarray
            Input feature array
        sensitive_idxs : List[int]
            Indices of sensitive attributes in the input array
        **kwargs : dict
            Additional parameters (unused)

        Returns
        -------
        DisparateImpactRemover
            Self, with fitted parameters
        """
        # Extract sensitive values array
        sensitive_col = inputs[:, sensitive_idxs[0]]
        self.sensitive_values = np.unique(sensitive_col.flatten())

        if self.sensitive_values is None:
            raise ValueError(
                "sensitive_values not set. _fit must be called before using sensitive_values."
            )

        self.quantile_maps.clear()
        n_features = inputs.shape[1]

        for feat_idx in range(n_features):
            # Group raw values
            groups = {
                g: inputs[sensitive_col.flatten() == g, feat_idx]
                for g in self.sensitive_values
            }
            # Common rank grid
            min_n = min(len(v) for v in groups.values())
            rank_common = np.linspace(0, 1, min_n)
            # Median values at each rank
            median_vals = np.array(
                [
                    np.median([np.quantile(vals, r) for vals in groups.values()])
                    for r in rank_common
                ]
            )
            # Store CDFs
            group_cdfs = {g: _make_cdf(v) for g, v in groups.items()}

            self.quantile_maps[feat_idx] = {
                "rank_common": rank_common,
                "median_vals": median_vals,
                "group_cdfs": group_cdfs,
            }

        return self

    def transform(self, inputs: np.ndarray, sensitive_idxs: List[int]) -> np.ndarray:
        """
        Apply the repair transformation to raw input array.

        Parameters
        ----------
        inputs : np.ndarray
            Input feature array to transform
        sensitive_idxs : List[int]
            Indices of sensitive attributes in the input array

        Returns
        -------
        np.ndarray
            Transformed feature array with repaired values
        """

        if self.sensitive_values is None:
            raise ValueError(
                "DisparateImpactRemover must be fitted before calling transform."
            )

        sensitive_col = inputs[:, sensitive_idxs[0]]
        X_out = inputs.copy()
        n_features = inputs.shape[1]

        for feat_idx in range(n_features):
            info = self.quantile_maps[feat_idx]
            r_common = info["rank_common"]
            m_vals = info["median_vals"]

            for g in self.sensitive_values:
                mask = sensitive_col.flatten() == g
                if not mask.any():
                    continue
                vals = inputs[mask, feat_idx]
                # ranks per value
                ranks = np.array([info["group_cdfs"][g](v) for v in vals])
                # map to median distribution
                repaired = np.interp(ranks, r_common, m_vals)
                # blend
                X_out[mask, feat_idx] = (
                    1 - self.repair_level
                ) * vals + self.repair_level * repaired

        return X_out
