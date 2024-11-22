from sklearn.base import BaseEstimator
from numpy import ndarray
from pandas import DataFrame
from typing import Union, Optional


class InProcessing(BaseEstimator):

    def __unpack_dataframe(
        self, x: Union[ndarray, DataFrame], y: Optional[ndarray] = None
    ) -> tuple[ndarray, Optional[ndarray]]:
        if isinstance(x, ndarray):
            if y is None:
                raise ValueError("y must be provided if x is a numpy array")
            return x, y
        target_columns = x.targets
        non_target_columns = x.columns.difference(target_columns)
        y = x[target_columns].values
        x = x[non_target_columns].values
        return x, y

    def fit(self, x, y):
        x, y = self.__unpack_dataframe(x, y)
        return self._fit(x, y)

    def _fit(self, x, y):
        raise NotImplementedError

    def transform(self, x, y):
        x, y = self.__unpack_dataframe(x, y)
        return self._transform(x, y)

    def _transform(self, x, y):
        raise NotImplementedError

    def fit_transform(self, x, y):
        x, y = self.__unpack_dataframe(x, y)
        return self._fit_transform(x, y)

    def _fit_transform(self, x, y):
        raise NotImplementedError
