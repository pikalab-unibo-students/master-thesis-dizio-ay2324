from fairlib.processing import DataFrameAwareTransformer

__all__ = ["Reweighing", "ReweighingWithMean"]

def _reweighing(privileged, unprivileged, favorable, unfavorable, n_total):
    n_favorable = favorable.sum()
    n_unfavorable = unfavorable.sum()

    n_privileged = privileged.sum()
    n_unprivileged = unprivileged.sum()

    n_privileged_favorable = (privileged & favorable).sum()
    n_privileged_unfavorable = (privileged & unfavorable).sum()
    n_unprivileged_favorable = (unprivileged & favorable).sum()
    n_unprivileged_unfavorable = (unprivileged & unfavorable).sum()

    weight_privileged_favorable = (
        (n_favorable * n_privileged) / (n_total * n_privileged_favorable)
        if n_privileged_favorable > 0
        else 0
    )
    weight_privileged_unfavorable = (
        (n_unfavorable * n_privileged) / (n_total * n_privileged_unfavorable)
        if n_privileged_unfavorable > 0
        else 0
    )
    weight_unprivileged_favorable = (
        (n_favorable * n_unprivileged) / (n_total * n_unprivileged_favorable)
        if n_unprivileged_favorable > 0
        else 0
    )
    weight_unprivileged_unfavorable = (
        (n_unfavorable * n_unprivileged) / (n_total * n_unprivileged_unfavorable)
        if n_unprivileged_unfavorable > 0
        else 0
    )

    return (weight_privileged_favorable,
            weight_privileged_unfavorable,
            weight_unprivileged_favorable,
            weight_unprivileged_unfavorable)


class Reweighing(DataFrameAwareTransformer):
    def transform(self, df, favorable_label=1):
        if len(df.targets) > 1:
            raise ValueError(
                "More than one “target” column is present. Reweighing supports only 1 target."
            )

        target_column = df.targets.pop()

        favorable = df[target_column] == favorable_label
        unfavorable = df[target_column] != favorable_label

        n_total = len(df)

        df["weights"] = 1.0

        privileged = []
        unprivileged = []

        for sensitive_column in df.sensitive:
            privileged = df[sensitive_column] == 1
            unprivileged = df[sensitive_column] == 0

        (weight_privileged_favorable,
         weight_privileged_unfavorable,
         weight_unprivileged_favorable,
         weight_unprivileged_unfavorable) = _reweighing(
            privileged, unprivileged, favorable, unfavorable, n_total)

        df.loc[privileged & favorable, "weights"] = weight_privileged_favorable
        df.loc[privileged & unfavorable, "weights"] = weight_privileged_unfavorable
        df.loc[unprivileged & favorable, "weights"] = weight_unprivileged_favorable
        df.loc[unprivileged & unfavorable, "weights"] = weight_unprivileged_unfavorable

        return df


class ReweighingWithMean(DataFrameAwareTransformer):
    def transform(self, df, favorable_label=1, remove_weight_columns=True):
        if len(df.targets) > 1:
            raise ValueError(
                "More than one “target” column is present. Reweighing supports only 1 target"
            )

        target_column = df.targets.pop()

        favorable = df[target_column] == favorable_label
        unfavorable = df[target_column] != favorable_label

        n_total = len(df)

        df["weights"] = 0.0

        weight_columns = []
        for sensitive_column in df.sensitive:
            privileged = df[sensitive_column] == 1
            unprivileged = df[sensitive_column] == 0

            (weight_privileged_favorable,
             weight_privileged_unfavorable,
             weight_unprivileged_favorable,
             weight_unprivileged_unfavorable) = _reweighing(
                privileged, unprivileged, favorable, unfavorable, n_total)

            weight_col_name = f"weights_{sensitive_column}"
            df[weight_col_name] = 1.0

            df.loc[privileged & favorable, weight_col_name] = weight_privileged_favorable
            df.loc[privileged & unfavorable, weight_col_name] = weight_privileged_unfavorable
            df.loc[unprivileged & favorable, weight_col_name] = weight_unprivileged_favorable
            df.loc[unprivileged & unfavorable, weight_col_name] = weight_unprivileged_unfavorable

            weight_columns.append(weight_col_name)

        df["weights"] = df[weight_columns].mean(axis=1)

        if remove_weight_columns:
            df.drop(columns=weight_columns, inplace=True)

        return df
