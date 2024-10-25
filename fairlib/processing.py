from numpy import ndarray
from pandas import DataFrame
from typing import Union, Optional, Protocol, runtime_checkable


@runtime_checkable
class Estimator(Protocol):
    def fit(self, *args, **kwargs): ...


@runtime_checkable
class Predictor(Protocol):
    def predict(self, *args, **kwargs): ...


@runtime_checkable
class Transformer(Protocol):
    def transform(self, *args, **kwargs): ...


@runtime_checkable
class FittableTransformer(Protocol):
    def fit(self, *args, **kwargs): ...
    def transform(self, *args, **kwargs): ...
    def fit_transform(self, *args, **kwargs): ...


@runtime_checkable
class Model(Protocol):
    def score(self, *args, **kwargs): ...


Processor = Union[Estimator, Predictor, Transformer, FittableTransformer, Model]
setattr(Processor, "__name__", "Processor")


def ensure_is_processor(obj):
    if not isinstance(obj, Processor):
        raise TypeError("Object must be an instance of Estimator, Predictor, Transformer, FittableTransformer, "
                        "or Model.")


def unpack_dataframe(
        x: Union[ndarray, DataFrame],
        y: Optional[ndarray] = None) -> tuple[ndarray, Optional[ndarray]]:
    if isinstance(x, ndarray):
        if y is None:
            raise ValueError("y must be provided if x is a numpy array")
        return x, y
    target_columns = list(x.targets)
    non_target_columns = x.columns.difference(target_columns)
    y = x[target_columns].values
    x = x[non_target_columns].values
    return x, y


class DataFrameAwareEstimator(Estimator):
    def fit(self, x, y=None):
        x, y = unpack_dataframe(x, y)
        return self._fit(x, y)


class DataFrameAwarePredictor(Predictor):
    def predict(self, x):
        x, _ = unpack_dataframe(x)
        return self._predict(x)


class DataFrameAwareTransformer(Transformer):
    def transform(self, x, y=None):
        x, y = unpack_dataframe(x, y)
        return self._transform(x, y)


class DataFrameAwareFittableTransformer(FittableTransformer, DataFrameAwareEstimator, DataFrameAwareTransformer):
    def fit_transform(self, x, y=None):
        x, y = unpack_dataframe(x, y)
        return self._fit_transform(x, y)


class DataFrameAwareModel(Model):
    def score(self, x, y=None):
        x, y = unpack_dataframe(x, y)
        return self._score(x, y)


class DataFrameAwareProcessorWrapper:
    def __init__(self, processor):
        self.__processor = self._initialize_processor(processor)

    def _initialize_processor(self, processor):
        ensure_is_processor(processor)
        return processor

    @property
    def processor(self):
        return self.__processor

    def _fit(self, *args, **kwargs):
        return self.__processor.fit(*args, **kwargs)

    def _predict(self, *args, **kwargs):
        return self.__processor.predict(*args, **kwargs)

    def _transform(self, *args, **kwargs):
        return self.__processor.transform(*args, **kwargs)

    def _fit_transform(self, *args, **kwargs):
        return self.__processor.fit_transform(*args, **kwargs)

    def _score(self, *args, **kwargs):
        return self.__processor.score(*args, **kwargs)