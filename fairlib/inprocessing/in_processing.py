from sklearn.base import BaseEstimator
from numpy import ndarray
from pandas import DataFrame


class InProcessingBase(BaseEstimator):
    def __unpack_dataframe( 
            self, 
            X: ndarray | DataFrame, 
            y: ndarray = None) -> tuple[ndarray, ndarray]:
        if isinstance(X, ndarray):
            if y is None:
                raise ValueError("y must be provided if X is a numpy array")
            return X, y
        target_columns = X.targets
        non_target_columns = X.columns.difference(target_columns)
        y = X[target_columns].values
        X = X[non_target_columns].values
        return X, y

    def fit(self, X, y):
        X, y = self.__unpack_dataframe(X, y)
        return self._fit(X, y)
    
    def _fit(self, X, y):
        raise NotImplementedError

    def transform(self, X, y):
        X, y = self.__unpack_dataframe(X, y)
        return self._transform(X, y)
    
    def _transform(self, X, y):
        raise NotImplementedError

    def fit_transform(self, X, y):
        X, y = self.__unpack_dataframe(X, y)
        return self._fit_transform(X, y)
    
    def _fit_transform(self, X, y):
        raise NotImplementedError
