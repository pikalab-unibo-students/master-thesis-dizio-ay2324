import pandas as pd
from fairlib.dataframe import DataFrameExtensionFunction


class PreProcessing:
    """
    Base class for applying pre-processing techniques on a dataset.

    Methods
    --------
    __call__(df):
        Should be implemented in subclasses to apply a pre-processing operation on a DataFrame.

    apply(name):
        Applies the extension function on a DataFrame using a function name.
    """

    def __call__(self, df):
        raise NotImplementedError

    def apply(self, name):
        DataFrameExtensionFunction(callable=self).apply(name=name)


class Reweighing(PreProcessing):
    """
    Implements the Reweighing method to balance privileged and unprivileged groups in a dataset.

    Methods
    --------
    __call__(df, favorable_label=1):
        Applies reweighing to the DataFrame and returns the DataFrame with a weight column. For multiple sensitive columns,
        it applies reweighing in a cumulative way, calculating the intersection of privileged and unprivileged groups
        across all sensitive attributes.

    Parameters
    -----------
    df : fairlearn.DataFrame (pandas.DataFrame)
        The dataset containing both sensitive and target columns.
    favorable_label : int, optional, default=1
        The value labeled as "favorable" in the target column.

    Returns
    --------
    fairlearn.DataFrame (pandas.DataFrame)
        A DataFrame with an additional 'weights' column containing the calculated weights for each example.

    Raises
    -------
    ValueError:
        If more than one target column is present in the dataset.
    """

    def __call__(self, df, favorable_label=1):
        if len(df.targets) > 1:
            raise ValueError(
                "More than one “target” column is present. Reweighing supports only 1 target."
            )

        target_column = df.targets.pop()

        privileged = pd.Series([True] * len(df))
        unprivileged = pd.Series([True] * len(df))

        for sensitive_column in df.sensitive:
            privileged &= df[sensitive_column] == 1
            unprivileged &= df[sensitive_column] == 0

        favorable = df[target_column] == favorable_label
        unfavorable = df[target_column] != favorable_label

        n_total = len(df)
        n_privileged = privileged.sum()
        n_unprivileged = unprivileged.sum()
        n_favorable = favorable.sum()
        n_unfavorable = unfavorable.sum()

        n_priv_fav = (privileged & favorable).sum()
        n_priv_unfav = (privileged & unfavorable).sum()
        n_unpriv_fav = (unprivileged & favorable).sum()
        n_unpriv_unfav = (unprivileged & unfavorable).sum()

        w_priv_fav = (
            (n_favorable * n_privileged) / (n_total * n_priv_fav)
            if n_priv_fav > 0
            else 0
        )
        w_priv_unfav = (
            (n_unfavorable * n_privileged) / (n_total * n_priv_unfav)
            if n_priv_unfav > 0
            else 0
        )
        w_unpriv_fav = (
            (n_favorable * n_unprivileged) / (n_total * n_unpriv_fav)
            if n_unpriv_fav > 0
            else 0
        )
        w_unpriv_unfav = (
            (n_unfavorable * n_unprivileged) / (n_total * n_unpriv_unfav)
            if n_unpriv_unfav > 0
            else 0
        )

        df["weights"] = 1.0

        df.loc[privileged & favorable, "weights"] = w_priv_fav
        df.loc[privileged & unfavorable, "weights"] = w_priv_unfav
        df.loc[unprivileged & favorable, "weights"] = w_unpriv_fav
        df.loc[unprivileged & unfavorable, "weights"] = w_unpriv_unfav

        return df


class ReweighingWithMean(PreProcessing):
    """
    Extends the Reweighing method to support multiple sensitive columns and computes the average of the weights.

    Methods
    --------
    __call__(df, favorable_label=1, remove_weight_columns=True):
        Applies reweighing to multiple sensitive columns and returns a DataFrame with an average weight column.

    Parameters
    -----------
    df : fairlearn.DataFrame (pandas.DataFrame)
        The dataset containing both sensitive and target columns.
    favorable_label : int, optional, default=1
        The value labeled as "favorable" in the target column.
    remove_weight_columns : bool, optional, default=True
        If set to True, removes the individual weight columns for each sensitive attribute after averaging.

    Returns
    --------
    fairlearn.DataFrame (pandas.DataFrame)
        A DataFrame with a 'weights' column containing the average weights for each example.

    Raises
    -------
    ValueError:
        If more than one target column is present in the dataset.
    """

    def __call__(self, df, favorable_label=1, remove_weight_columns=True):
        if len(df.targets) > 1:
            raise ValueError(
                "More than one “target” column is present. Reweighing supports only 1 target"
            )

        target_column = df.targets.pop()

        favorable = df[target_column] == favorable_label
        unfavorable = df[target_column] != favorable_label

        n_total = len(df)
        n_favorable = favorable.sum()
        n_unfavorable = unfavorable.sum()

        df["weights"] = 0.0

        weight_columns = []
        for sensitive_column in df.sensitive:
            privileged = df[sensitive_column] == 1
            unprivileged = df[sensitive_column] == 0

            n_privileged = privileged.sum()
            n_unprivileged = unprivileged.sum()

            n_priv_fav = (privileged & favorable).sum()
            n_priv_unfav = (privileged & unfavorable).sum()
            n_unpriv_fav = (unprivileged & favorable).sum()
            n_unpriv_unfav = (unprivileged & unfavorable).sum()

            w_priv_fav = (
                (n_favorable * n_privileged) / (n_total * n_priv_fav)
                if n_priv_fav > 0
                else 0
            )
            w_priv_unfav = (
                (n_unfavorable * n_privileged) / (n_total * n_priv_unfav)
                if n_priv_unfav > 0
                else 0
            )
            w_unpriv_fav = (
                (n_favorable * n_unprivileged) / (n_total * n_unpriv_fav)
                if n_unpriv_fav > 0
                else 0
            )
            w_unpriv_unfav = (
                (n_unfavorable * n_unprivileged) / (n_total * n_unpriv_unfav)
                if n_unpriv_unfav > 0
                else 0
            )

            weight_col_name = f"weights_{sensitive_column}"
            df[weight_col_name] = 1.0

            df.loc[privileged & favorable, weight_col_name] = w_priv_fav
            df.loc[privileged & unfavorable, weight_col_name] = w_priv_unfav
            df.loc[unprivileged & favorable, weight_col_name] = w_unpriv_fav
            df.loc[unprivileged & unfavorable, weight_col_name] = w_unpriv_unfav

            weight_columns.append(weight_col_name)

        df["weights"] = df[weight_columns].mean(axis=1)

        if remove_weight_columns:
            df.drop(columns=weight_columns, inplace=True)

        return df


Reweighing().apply("reweighing")
ReweighingWithMean().apply("reweighing_with_mean")
