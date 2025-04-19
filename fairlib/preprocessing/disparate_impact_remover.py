import numpy as np
import pandas as pd


class DisparateImpactRemover:
    """
    Fairness-aware repair algorithm that removes disparate impact
    by transforming feature distributions to a common (median) distribution
    across sensitive attribute groups, as described in Feldman et al. (2015).

    Parameters
    ----------
    repair_level : float, default=1.0
        Degree of repair.
        - 1.0 = full repair (max fairness, min predictability of sensitive attribute)
        - 0.0 = no repair (original data)
    """

    def __init__(self, repair_level=1.0):
        self.repair_level = repair_level
        self.feature_names = None
        self.sensitive_values = None
        self.quantile_maps = {}

    def fit(self, X, y=None, s=None, **kwargs):
        """
        Fit the repair model: compute per-group CDFs and
        the median quantile distribution for each feature.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features (numeric only).
        s : array-like
            Sensitive attribute values (e.g., gender, race).
        """
        if s is None:
            raise ValueError("DisparateImpactRemover requires the sensitive attribute 's'")

        s = np.array(s).flatten()
        self.sensitive_values = np.unique(s)

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        else:
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        self.quantile_maps = {}

        for feat_idx in range(X.shape[1]):
            # Step 1: Collect values by sensitive group
            values_by_group = {
                group: X[s == group, feat_idx]
                for group in self.sensitive_values
            }

            # Step 2: Compute empirical quantiles for each group
            quantiles = {}
            for group, values in values_by_group.items():
                sorted_vals = np.sort(values)
                ranks = np.linspace(0, 1, len(sorted_vals))
                quantiles[group] = (ranks, sorted_vals)

            # Step 3: Create a shared "median" quantile function across groups
            # At each rank, the value is the median of the group-specific quantile values
            median_vals = []
            rank_common = np.linspace(0, 1, min(len(v) for v in values_by_group.values()))
            for r in rank_common:
                group_qvals = [np.quantile(values_by_group[g], r) for g in self.sensitive_values]
                median_vals.append(np.median(group_qvals))
            median_vals = np.array(median_vals)

            # Store mapping information for later transformation
            self.quantile_maps[feat_idx] = {
                'rank_common': rank_common,
                'median_vals': median_vals,
                'group_cdfs': {
                    g: self._make_cdf(values_by_group[g]) for g in self.sensitive_values
                }
            }
        return self

    def transform(self, X, **kwargs):
        """
        Apply the fairness repair transformation to the input data.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Input features to transform.
        s : array-like (required in kwargs)
            Sensitive attribute values for each row.

        Returns
        -------
        Transformed X (same type as input) with reduced disparate impact.
        """
        if 's' not in kwargs:
            raise ValueError("DisparateImpactRemover.transform requires the sensitive attribute 's'")

        s = np.array(kwargs['s']).flatten()

        if isinstance(X, pd.DataFrame):
            X_values = X.values
            is_dataframe = True
        else:
            X_values = X
            is_dataframe = False

        X_transformed = X_values.copy()

        for feat_idx in range(X_values.shape[1]):
            map_info = self.quantile_maps[feat_idx]
            rank_common = map_info['rank_common']
            median_vals = map_info['median_vals']

            for group in self.sensitive_values:
                mask = (s == group)
                if not np.any(mask):
                    continue

                group_vals = X_values[mask, feat_idx]
                cdf_func = map_info['group_cdfs'][group]

                # Step 1: Compute empirical rank (percentile) for each value
                ranks = np.array([cdf_func(val) for val in group_vals])

                # Step 2: Use rank to find corresponding value in median distribution
                repaired_vals = np.interp(ranks, rank_common, median_vals)

                # Step 3: Apply partial repair (interpolate between original and repaired)
                final_vals = (1 - self.repair_level) * group_vals + self.repair_level * repaired_vals
                X_transformed[mask, feat_idx] = final_vals

        if is_dataframe:
            return pd.DataFrame(X_transformed, columns=self.feature_names, index=X.index)
        return X_transformed

    def fit_transform(self, X, y=None, s=None, **kwargs):
        """
        Convenience method: fit and transform in one step.
        """
        return self.fit(X, y=y, s=s, **kwargs).transform(X, s=s)

    def _make_cdf(self, values):
        """
        Create an empirical cumulative distribution function (CDF)
        based on sorted values.
        """
        sorted_vals = np.sort(values)

        def cdf(val):
            return np.searchsorted(sorted_vals, val, side='right') / len(sorted_vals)

        return cdf