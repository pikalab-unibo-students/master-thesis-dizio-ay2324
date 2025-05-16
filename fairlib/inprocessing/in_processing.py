from fairlib import DataFrame
from typing import Optional, Any


class Processor:

    def __init__(self, *args, **kwargs):
        pass

    def fit(self, x: DataFrame, y: Optional[Any] = None, **kwargs):
        """
        Fit the processor to the data.

        Parameters
        ----------
        x : DataFrame
            Input features to fit.
        y : Series or None, optional
            Target labels (not used in this base class).
        **kwargs : Additional keyword arguments for specific processors.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, x: DataFrame, **kwargs) -> DataFrame:
        """
        Predict using the processor.

        Parameters
        ----------
        x : DataFrame
            Input features to predict.
        **kwargs : Additional keyword arguments for specific processors.

        Returns
        -------
        DataFrame
            Predicted values.
        """
        raise NotImplementedError("Subclasses should implement this method.")
