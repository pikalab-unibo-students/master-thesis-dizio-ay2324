from typing import Optional, Any, TypeVar, Generic
from fairlib import DataFrame

T = TypeVar("T")


class Preprocessor(Generic[T]):
    """
    Base class for preprocessing algorithms.

    All fairness preprocessing algorithms should inherit from this class
    and implement the fit_transform method.
    """

    def __init__(self, *args, **kwargs):
        """Initialize the preprocessor with optional arguments."""
        pass

    def fit(self, X: DataFrame, y: Optional[Any] = None, **kwargs) -> "Preprocessor":
        """
        Fit the preprocessor to the data.

        Parameters
        ----------
        X : DataFrame
            Input features to fit.
        y : Series or None, optional
            Target labels (not used in most preprocessing algorithms).
        **kwargs : dict
            Additional algorithm-specific parameters.

        Returns
        -------
        Preprocessor
            The fitted preprocessor instance.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def transform(self, X: DataFrame) -> DataFrame:
        """
        Transform the input data using the fitted preprocessor.

        Parameters
        ----------
        X : DataFrame
            Input features to transform.

        Returns
        -------
        DataFrame
            Transformed data (typically same type as input).
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def fit_transform(self, X: DataFrame, y: Optional[Any] = None, **kwargs) -> T:
        """
        Fit the preprocessor to the data and transform it.

        Parameters
        ----------
        X : DataFrame
            Input features to fit and transform.
        y : Series or None, optional
            Target labels (not used in most preprocessing algorithms).
        **kwargs : dict
            Additional algorithm-specific parameters.

        Returns
        -------
        T
            Transformed data (typically same type as input).
        """
        raise NotImplementedError("Subclasses must implement this method.")
