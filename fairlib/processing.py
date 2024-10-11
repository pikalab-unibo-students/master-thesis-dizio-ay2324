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
        raise TypeError("Object must be an instance of Estimator, Predictor, Transformer, FittableTransformer, or Model.")


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
        ensure_is_processor(processor)
        self.__processor = self._initialize_processor(processor)

    def _initialize_processor(self, processor):
        return processor

    @property
    def processor(self):
        return self._processor

    def _fit(self, x, y):
        return self.__processor.fit(x, y)       
     
    def _predict(self, x):
        return self.__processor.predict(x)  
              
    def _transform(self, x, y):
        return self.__processor.transform(x, y) 
               
    def _fit_transform(self, x, y):
        return self.__processor.fit_transform(x, y)   
             
    def _score(self, x, y):
        return self.__processor.score(x, y)            
    

# TODO turn this into unit tests and remove
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from fairlib import DataFrame
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier

    processor1 = StandardScaler()
    processor2 = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)

    for processor in [processor1, processor2]:
        for protocol in [Estimator, Predictor, Transformer, FittableTransformer, Model, Processor]:
            print(f"{type(processor).__name__} is instance of {protocol.__name__}: {isinstance(processor, protocol)}")

    iris = load_iris()
    df = DataFrame(data=iris.data, columns=iris.feature_names)
    df['Class'] = iris.target
    df.targets = 'Class'
    iris = df
    try:
        processor2.fit(iris)
    except TypeError as e:
        print(e)
    
    class DataFrameAwareMLPClassifier(DataFrameAwareProcessorWrapper, DataFrameAwareEstimator, DataFrameAwarePredictor, DataFrameAwareModel):
        def __init__(self, mlp):
            assert isinstance(mlp, MLPClassifier)
            super().__init__(mlp)
    
    processor2 = DataFrameAwareMLPClassifier(processor2)
    processor2.fit(iris)
