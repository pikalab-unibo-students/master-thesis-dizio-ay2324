from fairlib import DataFrameExtensionFunction
from pandas.api.types import is_numeric_dtype
import pandas as pd


class PreProcessing:
    def __call__(self, df):
        raise NotImplementedError

    def apply(self, name):
        DataFrameExtensionFunction(callable=self).apply(name=name)


class Reweighing(PreProcessing):
    def __call__(self, df, favorable_label=1):
        if len(df.targets) > 1:
            raise ValueError("More than one “target” column is present. Reweighing supports only 1 target.")

        target_column = df.targets.pop()
        
        privileged = pd.Series([True] * len(df)) 
        unprivileged = pd.Series([True] * len(df))
        
        for sensitive_column in df.sensitive:
            privileged &= (df[sensitive_column] == 1)
            unprivileged &= (df[sensitive_column] == 0)
        
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

        w_priv_fav = (n_favorable * n_privileged) / (n_total * n_priv_fav) if n_priv_fav > 0 else 0
        w_priv_unfav = (n_unfavorable * n_privileged) / (n_total * n_priv_unfav) if n_priv_unfav > 0 else 0
        w_unpriv_fav = (n_favorable * n_unprivileged) / (n_total * n_unpriv_fav) if n_unpriv_fav > 0 else 0
        w_unpriv_unfav = (n_unfavorable * n_unprivileged) / (n_total * n_unpriv_unfav) if n_unpriv_unfav > 0 else 0

        df['weights'] = 1.0

        df.loc[privileged & favorable, 'weights'] = w_priv_fav
        df.loc[privileged & unfavorable, 'weights'] = w_priv_unfav
        df.loc[unprivileged & favorable, 'weights'] = w_unpriv_fav
        df.loc[unprivileged & unfavorable, 'weights'] = w_unpriv_unfav

        return df
    
        
        
Reweighing().apply("reweighing")


