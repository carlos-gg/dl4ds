from scipy import sparse
import numpy as np
import xarray as xr
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
    where min, max = value_range.
    
    This transformation is often used as an alternative to zero mean,
    unit variance scaling.
    
    Parameters
    ----------
    value_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.
    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).
    axis : int, tuple of int or None, default=None
        Axis or axes along which the minimum and maximum will be computed (via 
        ``np.nanmin`` and ``np.nanmax`` functions). If None then the new range
        is computed from the whole dataset (all dimensions/axes).
    fillnanto : float or int, deafult=-1
        Value to be used when filling in NaN values. 
    
    Notes
    -----
    NaNs are disregarded in fit when transforming to the new value range, and 
    then replaced according to ``fillnanto`` in transform. 
    """

    def __init__(self, value_range=(0, 1), copy=True, axis=None, fillnanto=-1):
        self.value_range = value_range
        self.copy = copy
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
            del self.data_min_
            del self.data_max_
            del self.data_range_

    def fit(self, X, y=None):
        """Calculate the minimum and maximum to be used for later scaling.

        Parameters
        ----------
        X : xr.DataArray or np.ndarray
            The data used to compute the minimum and maximum used for later 
            scaling along the desired axis.
        y : None
            Ignored.
        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Calculate the min and max on X for later scaling.

        Parameters
        ----------
        X : xr.DataArray or np.ndarray
            The data used to compute the minimum and maximum used for later 
            scaling along the desired axis.
        y : None
            Ignored.
        """
        X = np.squeeze(X)
        value_range = self.value_range
        if value_range[0] >= value_range[1]:
            raise ValueError("Minimum of desired value_range must be smaller than maximum. Got %s."% str(range))

        if sparse.issparse(X):
            raise TypeError("MinMaxScaler does not support sparse input.")
        
        ### creating a nan mask
        if np.any(np.isnan(X)):
            self.nan_mask = np.isnan(X)

        ### data type validation
        if isinstance(X, np.ndarray):
            data_min = np.nanmin(X, axis=self.axis, keepdims=True)
            data_max = np.nanmax(X, axis=self.axis, keepdims=True)
        elif isinstance(X, xr.DataArray):
            data_min = X.min(axis=self.axis, skipna=True, keepdims=True).values
            data_max = X.max(axis=self.axis, skipna=True, keepdims=True).values
        else:
            raise TypeError('`X` is neither a np.ndarray or xr.DataArray')

        data_range = data_max - data_min
        self.scale_ = (value_range[1] - value_range[0]) / _handle_zeros_in_scale(
            data_range, copy=True)
        self.min_ = value_range[0] - data_min * self.scale_
        self.data_min_ = data_min
        self.data_max_ = data_max
        self.data_range_ = data_range
        return self

    def transform(self, X):
        """Scale X according to range.

        Parameters
        ----------
        X : xr.DataArray or np.ndarray
            Input data that will be transformed.
        """
        check_is_fitted(self)
        X = np.squeeze(X)

        if self.copy:
            X = X.copy()

        X *= self.scale_
        X += self.min_
            
        ### filling nan values
        if np.any(np.isnan(X)):
            if isinstance(X, np.ndarray):
                X = np.nan_to_num(X, nan=self.fillnanto)
            elif isinstance(X, xr.DataArray):
                X = X.fillna(value=self.fillnanto)
        return X

    def inverse_transform(self, X):
        """Undo the scaling of X according to range.

        Parameters
        ----------
        X : xr.DataArray or np.ndarray
            Input data that will be transformed.
        """
        check_is_fitted(self)
        X = np.squeeze(X)

        if self.copy:
            X = X.copy()

        ### restoring nan mask
        if hasattr(self, 'nan_mask'):
            if isinstance(X, np.ndarray):
                X[self.nan_mask] = np.nan
            elif isinstance(X, xr.DataArray):
                X.values[self.nan_mask] = np.nan

        X -= self.min_
        X /= self.scale_
        return X

    def _more_tags(self):
        return {"allow_nan": True}


class StandardScaler(TransformerMixin, BaseEstimator):
    """Standardize features by removing the mean and scaling to unit variance.
    
    The standard score of a sample `x` is calculated as:
        z = (x - u) / s
    where `u` is the mean of the data or zero if `with_mean=False`,
    and `s` is the standard deviation of the data or one if `with_std=False`.
    
    Mean and standard deviation are then stored to be used on later data using
    :meth:`transform`.
    
    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.
    with_mean : bool, default=True
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.
    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).
    axis : None or int or tuple of int, default=None
        Axis or axes along which the minimum and maximum will be computed (via 
        ``np.nanmin`` and ``np.nanmax`` functions). If None then the new range
        is computed from the whole dataset (all dimensions/axes).
    fillnanto : float or int, deafult=0
        Value to be used when filling in NaN values. 

    Notes
    -----
    NaNs are disregarded in fit when transforming to the new value range, and 
    then replaced according to ``fillnanto`` in transform. 
    """

    def __init__(self, copy=True, with_mean=True, with_std=True, axis=None,
                 fillnanto=0):
        self.with_mean = with_mean
        self.with_std = with_std
        self.copy = copy
        self.axis = axis
        self.fillnanto = fillnanto

    def _reset(self):
        """Reset internal data-dependent state of the scaler, if necessary.
        __init__ parameters are not touched.
        """
        # Checking one attribute is enough, because they are all set together
        # in partial_fit
        if hasattr(self, "mean_"):
            del self.mean_
            del self.std_

    def fit(self, X, y=None):
        """Calculate the mean and standard deviation of X for later scaling.

        Parameters
        ----------
        X : xr.DataArray or np.ndarray
            The data used to compute the mean and standard deviation
            used for later scaling along the features axis.
        y : None
            Ignored.
        """
        # Reset internal state before fitting
        self._reset()
        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Calculate the mean and standard deviation of X for later scaling.

        Parameters
        ----------
        X : xr.DataArray or np.ndarray
            The data used to compute the mean and standard deviation
            used for later scaling along the desired axis.
        y : None
            Ignored.
        """
        X = np.squeeze(X)
        ### creating a nan mask
        if np.any(np.isnan(X)):
            self.nan_mask = np.isnan(X)

        ### data type validation
        if isinstance(X, np.ndarray):
            if self.with_mean:
                data_mean = np.nanmean(X, axis=self.axis, keepdims=True)
            if self.with_std:
                data_std = np.nanstd(X, axis=self.axis, keepdims=True)
        elif isinstance(X, xr.DataArray):
            if self.with_mean:
                data_mean = X.mean(axis=self.axis, skipna=True, keepdims=True).values
            if self.with_std:
                data_std = X.std(axis=self.axis, skipna=True, keepdims=True).values
        else:
            raise TypeError('`X` is neither a np.ndarray or xr.DataArray')

        if self.with_mean:
            self.mean_ = data_mean
        if self.with_std:
            self.std_ = data_std
        return self

    def transform(self, X):
        """ Perform standardization by centering and scaling.

        Parameters
        ----------
        X : xr.DataArray or np.ndarray
            The data used to scale along the desired axis.
        """
        check_is_fitted(self)
        X = np.squeeze(X)

        if self.copy:
            X = X.copy()

        if self.with_std:
            X -= self.mean_
        if self.with_std:
            X /= self.std_
            
        ### filling nan values
        if np.any(np.isnan(X)):
            if isinstance(X, np.ndarray):
                X = np.nan_to_num(X, nan=self.fillnanto)
            elif isinstance(X, xr.DataArray):
                X = X.fillna(value=self.fillnanto)
        return X

    def inverse_transform(self, X):
        """Scale back the data to the original representation.

        Parameters
        ----------
        X : xr.DataArray or np.ndarray
            The data used to scale along the desired axis.
        """
        check_is_fitted(self)
        X = np.squeeze(X)

        if self.copy:
            X = X.copy()

        ### restoring nan mask
        if hasattr(self, 'nan_mask'):
            if isinstance(X, np.ndarray):
                X[self.nan_mask] = np.nan
            elif isinstance(X, xr.DataArray):
                X.values[self.nan_mask] = np.nan

        if self.with_std:
            X *= self.std_
        if self.with_mean:
            X += self.mean_
        return X

    def _more_tags(self):
        return {"allow_nan": True}