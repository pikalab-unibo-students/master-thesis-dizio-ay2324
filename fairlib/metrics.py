from fairlib import DataFrame, DataFrameExtensionFunction
from pandas.api.types import is_numeric_dtype
from pandas import concat as concat_df

class Metric:
    def __call__(self, df):
        raise NotImplementedError
    
    def apply(self, name):
        DataFrameExtensionFunction(callable=self).apply(name=name)


class StatsMetric(Metric):
    def __call__(self, df):
        res = DataFrame(columns = ['column', 'min', 'max', 'mean', 'std'])
        for column in df.columns:
            numeric = is_numeric_dtype(df[column])
            row = {
                'column': column,
                'min': df[column].min() if numeric else None,
                'max': df[column].max() if numeric else None,
                'mean': df[column].mean() if numeric else None,
                'std': df[column].std() if numeric else None,
            }
            res = concat_df([res, DataFrame([row])], ignore_index=True)
        return res
    

StatsMetric().apply('stats')
