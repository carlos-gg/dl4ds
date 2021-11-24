from scipy import sparse
import numpy as np
from sklearn.utils.validation import check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing._data import _handle_zeros_in_scale


class MinMaxScaler(TransformerMixin, BaseEstimator):
    """Transform data to a given range.
    This estimator scales and translates the data distribution such
    that it is in the given range on the training set, e.g. between
    zero and one. If NaN values are present there will be replaced by a given
    value, e.g. minus one. 

    The transformation is given by::
        X_std = (X - X.min(axis)) / (X.max(axis) - X.min(axis))
        X_scaled = X_std * (max - min) + min
    where min, max = range.
    
    This transformation is often used as an alternative to zero mean,
    unit variance scaling.
    
    Parameters
    ----------
    range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).
    clip : bool, default=False
        Set to True to clip transformed values of held-out data to
        provided `range`.
    axis : int, tuple of int or None, optional
        Axis or axes along which the minimum and maximum will be computed (via 
        ``np.nanmin`` and ``np.nanmax`` functions). If None then the new range
    fillnanto : float or int, optional
        Value to be used when filling in NaN values. 
    
    Notes
    -----
    NaNs are disregarded in fit when transforming to the new value range, and 
    then replaced according to ``fillnanto`` in transform. 
    
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

    def __init__(self, range=(0, 1), *, copy=True, clip=False, axis=None, 
                 fillnanto=-1):
        self.range = range
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
        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Online computation of min and max on X for later scaling.
        """
        range = self.range
        if range[0] >= range[1]:
            raise ValueError(
                "Minimum of desired feature range must be smaller than maximum. Got %s."
                % str(range)
            )

        if sparse.issparse(X):
            raise TypeError("MinMaxScaler does not support sparse input.")

        first_pass = not hasattr(self, "n_samples_seen_")
        
        ### TO-DO: other data validation steps 

        if self.copy:
            X = X.copy()
        
        ### creating a nan mask
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
        self.scale_ = (range[1] - range[0]) / _handle_zeros_in_scale(
            data_range, copy=True
        )
        self.min_ = range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        """Scale features of X according to range.
        """
        check_is_fitted(self)

        ### TO-DO: other data validation steps 
        
        X *= self.scale_
        X += self.min_
        if self.clip:
            np.clip(X, self.range[0], self.range[1], out=X)
            
        ### filling nan values
        if np.any(np.isnan(X)):
            X = np.nan_to_num(X, copy=self.copy, nan=self.fillnanto)
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to range.
        """
        check_is_fitted(self)
    
        ### TO-DO: other data validation steps 

        X = X.copy()

        ### restoring nan mask
        if hasattr(self, 'nan_mask'):
            X[self.nan_mask] = np.nan

        X -= self.min_
        X /= self.scale_
        return X

    def _more_tags(self):
        return {"allow_nan": True}