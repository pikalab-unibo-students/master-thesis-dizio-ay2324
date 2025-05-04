class Preprocessor:
    """
    Base class for preprocessing algorithms.
    """

    def fit_transform(self, X, y=None, **kwargs):
        """
        Fit the preprocessor to the data and transform it.

        Parameters
        ----------
        X : DataFrame
            Input features to fit and transform.
        y : Series or None, optional
            Target labels (not used in this base class).

        Returns
        -------
        Transformed X (same type as input).
        """
        raise NotImplementedError("Subclasses should implement this method.")