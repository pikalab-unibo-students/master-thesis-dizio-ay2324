from fairlib.dataframe import DataFrame
from fairlib.processing import *
import fairlib.keras as keras


class Fauci(DataFrameAwareProcessorWrapper, DataFrameAwareEstimator, DataFrameAwarePredictor, DataFrameAwareModel):
    def __init__(self, model: keras.Model, regularizer: Union[str, keras.losses.Loss], **kwargs):
        if not isinstance(model, keras.Model):
            raise TypeError(f"Expected a Keras model, got {type(model)}")
        super().__init__(model)
        self.__compilation_parameters = kwargs
        if regularizer is None:
            raise ValueError("Regularizer must be provided")
        self.__regularizer = regularizer

    def fit(self, x: DataFrame):
        if not isinstance(x, DataFrame):
            raise TypeError(f"Expected a DataFrame, got {type(x)}")
        sensitive_names = list(x.sensitive)
        if len(sensitive_names) != 1:
            raise ValueError(f"FaUCI expects exactly one sensitive column, got {len(sensitive_names)}: {sensitive_names}")
        sensitive_index: int = x.columns.get_loc(sensitive_names[0])
        x, y = unpack_dataframe(x)
        if y.shape[1] != 1:
            raise ValueError(f"FaUCI expects exactly one target column, got {y.shape[1]}")
        model: keras.Model = self.processor
        compilation_params = self.__compilation_parameters.copy()
        loss = ... # combine self._baseloss with self._regularizer(sensitive_index)
        compilation_params["loss"] = loss
        model.compile(**compilation_params)
        return self._fit(x, y)
    
    @property
    def _baseloss(self):
        loss = self.__compilation_parameters.get("loss")
        if isinstance(loss, str):
            return getattr(keras.losses, loss)
        return loss
    
    def _loss_function(self, sensitive_column_index: int) -> keras.losses.Loss:
        if isinstance(self.__regularizer, keras.losses.Loss):
            return self.__regularizer
        # TODO assume string, convert to loss accordingly


if __name__ == "__main__":
    algo = Fauci(model=keras.Sequential(...), regularizer="weighted_statistical_parity", optimizer="adam", loss="binary_crossentropy")