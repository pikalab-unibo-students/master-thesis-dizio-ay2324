import numpy as np
from typing import Any, Optional
from fairlib import DataFrame
from .pre_processing import Preprocessor


def _make_cdf(values: np.ndarray):
    """
    Create an empirical cumulative distribution function (CDF)
    based on sorted values.
    """
    sorted_vals = np.sort(values)

    def cdf(val: float) -> float:
        # Proportion of samples <= val
        return np.searchsorted(sorted_vals, val, side='right') / len(sorted_vals)

    return cdf


class DisparateImpactRemover(Preprocessor):
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
        self.repair_level = repair_level
        self.input_names: Optional[list[str]] = None
        self.target_names: Optional[list[str]] = None
        self.sensitive_names: Optional[list[str]] = None
        self.sensitive_values: Optional[np.ndarray] = None
        self.quantile_maps: dict[int, Any] = {}

    def fit_transform(self, X: DataFrame, y: Optional[Any] = None, **kwargs) -> DataFrame:
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
        if not isinstance(X, DataFrame):
            raise TypeError(f"Expected a fairlib DataFrame, got {type(X)}")

        # Unpack data
        inputs, _, input_names, target_names, sensitive_names, sensitive_indexes = X.unpack()

        if len(sensitive_indexes) != 1:
            raise ValueError(
                f"Expected exactly one sensitive column, got {len(sensitive_indexes)}"
            )
        self.input_names = input_names
        self.target_names = target_names
        self.sensitive_names = sensitive_names

        # Fit
        self._fit(inputs, sensitive_indexes, **kwargs)
        # Transform
        transformed = self._transform(inputs, sensitive_indexes)

        # Reconstruct DataFrame preserving metadata
        transformed_df = DataFrame(
            transformed,
            columns=self.input_names)

        transformed_df.sensitive = self.sensitive_names
        return transformed_df

    def _fit(
        self,
        inputs: np.ndarray,
        sensitive_idxs: list[int],
        **kwargs
    ) -> "DisparateImpactRemover":
        """
        Compute per-group CDFs and median quantile distributions.
        """
        # Extract sensitive values array
        sensitive_col = inputs[:, sensitive_idxs[0]]
        self.sensitive_values = np.unique(sensitive_col.flatten())

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
            median_vals = np.array([
                np.median([np.quantile(vals, r) for vals in groups.values()])
                for r in rank_common
            ])
            # Store CDFs
            group_cdfs = {g: _make_cdf(v) for g, v in groups.items()}

            self.quantile_maps[feat_idx] = {
                'rank_common': rank_common,
                'median_vals': median_vals,
                'group_cdfs': group_cdfs
            }

        return self

    def _transform(
        self,
        inputs: np.ndarray,
        sensitive_idxs: list[int]
    ) -> np.ndarray:
        """
        Apply the repair transformation to raw input array.
        """
        sensitive_col = inputs[:, sensitive_idxs[0]]
        X_out = inputs.copy()
        n_features = inputs.shape[1]

        for feat_idx in range(n_features):
            info = self.quantile_maps[feat_idx]
            r_common = info['rank_common']
            m_vals = info['median_vals']

            for g in self.sensitive_values:
                mask = sensitive_col.flatten() == g
                if not mask.any():
                    continue
                vals = inputs[mask, feat_idx]
                # ranks per value
                ranks = np.array([info['group_cdfs'][g](v) for v in vals])
                # map to median distribution
                repaired = np.interp(ranks, r_common, m_vals)
                # blend
                X_out[mask, feat_idx] = (
                    (1 - self.repair_level) * vals + self.repair_level * repaired
                )

        return X_out
