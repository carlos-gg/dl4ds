from scipy import sparse
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing._data import _handle_zeros_in_scale


class MinMaxScaler(TransformerMixin, BaseEstimator):
    """Transform features by scaling each feature to a given range.
    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.
    The transformation is given by::
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min
    where min, max = feature_range.
    This transformation is often used as an alternative to zero mean,
    unit variance scaling.
    Read more in the :ref:`User Guide <preprocessing_scaler>`.
    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).
    clip : bool, default=False
        Set to True to clip transformed values of held-out data to
        provided `feature range`.
        .. versionadded:: 0.24
    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Per feature adjustment for minimum. Equivalent to
        ``min - X.min(axis=0) * self.scale_``
    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data. Equivalent to
        ``(max - min) / (X.max(axis=0) - X.min(axis=0))``
        .. versionadded:: 0.17
           *scale_* attribute.
    data_min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the data
        .. versionadded:: 0.17
           *data_min_*
    data_max_ : ndarray of shape (n_features,)
        Per feature maximum seen in the data
        .. versionadded:: 0.17
           *data_max_*
    data_range_ : ndarray of shape (n_features,)
        Per feature range ``(data_max_ - data_min_)`` seen in the data
        .. versionadded:: 0.17
           *data_range_*
    n_features_in_ : int
        Number of features seen during :term:`fit`.
        .. versionadded:: 0.24
    n_samples_seen_ : int
        The number of samples processed by the estimator.
        It will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.
    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.
        .. versionadded:: 1.0
    See Also
    --------
    minmax_scale : Equivalent function without the estimator API.
    Notes
    -----
    NaNs are treated as missing values: disregarded in fit, and maintained in
    transform.
    For a comparison of the different scalers, transformers, and normalizers,
    see :ref:`examples/preprocessing/plot_all_scaling.py
    <sphx_glr_auto_examples_preprocessing_plot_all_scaling.py>`.
    Examples
    --------
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    >>> scaler = MinMaxScaler()
    >>> print(scaler.fit(data))
    MinMaxScaler()
    >>> print(scaler.data_max_)
    [ 1. 18.]
    >>> print(scaler.transform(data))
    [[0.   0.  ]
     [0.25 0.25]
     [0.5  0.5 ]
     [1.   1.  ]]
    >>> print(scaler.transform([[2, 2]]))
    [[1.5 0. ]]
    """

    def __init__(self, feature_range=(0, 1), *, copy=True, fillnanto=-1, 
                 clip=False, axis=None):
        self.feature_range = feature_range
        self.copy = copy
        self.clip = clip
        self.fillnanto = fillnanto
        self.axis = axis

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, "scale_"):
            del self.scale_
            del self.min_
            del self.n_samples_seen_
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X, y=None):
        """Compute the minimum and maximum to be used for later scaling.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the per-feature minimum and maximum
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Online computation of min and max on X for later scaling.
        All of X is processed as a single batch. This is intended for cases
        when :meth:`fit` is not feasible due to very large number of
        `n_samples` or because X is read from a continuous stream.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        Returns
        -------
        self : object
            Fitted scaler.
        """
        feature_range = self.feature_range
        if feature_range[0] >= feature_range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(feature_range)
            )

        if sparse.issparse(X):
            raise TypeError(
                "MinMaxScaler does not support sparse input. "
                "Consider using MaxAbsScaler instead."
            )

        first_pass = not hasattr(self, "n_samples_seen_")
        
        ### other data validation steps TO-DO
        X = X.copy()
        
        if np.any(np.isnan(X)):
            self.nan_mask = np.isnan(X)

        data_min = np.nanmin(X, axis=self.axis)
        data_max = np.nanmax(X, axis=self.axis)

        if first_pass:
            self.n_samples_seen_ = X.shape[0]
        else:
            data_min = np.minimum(self.data_min_, data_min)
            data_max = np.maximum(self.data_max_, data_max)
            self.n_samples_seen_ += X.shape[0]

        data_range = data_max - data_min
        self.scale_ = (feature_range[1] - feature_range[0]) / _handle_zeros_in_scale(
            data_range, copy=True
        )
        self.min_ = feature_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        """Scale features of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)

        ### other data validation steps TO-DO
        
        X *= self.scale_
        X += self.min_
        if self.clip:
            np.clip(X, self.feature_range[0], self.feature_range[1], out=X)
            
        ### filling nan values
        X = np.nan_to_num(X, self.fillnanto)
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to feature_range.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data that will be transformed. It cannot be sparse.
        Returns
        -------
        Xt : ndarray of shape (n_samples, n_features)
            Transformed data.
        """
        check_is_fitted(self)
    
        """
        X = check_array(
            X, copy=self.copy, dtype=FLOAT_DTYPES, force_all_finite="allow-nan"
        )"""
        ### restoring nan mask
        X[self.nan_mask] = np.nan

        X -= self.min_
        X /= self.scale_
        return X

    def _more_tags(self):
        return {"allow_nan": True}