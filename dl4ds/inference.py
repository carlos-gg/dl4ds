import os
import numpy as np
import xarray as xr
import tensorflow as tf

from .utils import (resize_array, spatial_to_temporal_samples, checkarray_ndim, 
                    Timing)
from . import SPATIAL_MODELS, SPATIOTEMP_MODELS, POSTUPSAMPLING_METHODS
from .dataloader import _get_season_, _get_season_array_


def predict(
    model, 
    data, 
    scale, 
    data_in_hr=True,
    use_season=True,
    topography=None, 
    landocean=None, 
    predictors=None, 
    time_window=None,
    interpolation='bicubic', 
    mean_std=None,
    save_path=None,
    save_fname='y_hat.npy',
    return_lr=False,
    stochastic_output=False,
    device='CPU'):
    """Inference with ``model`` on ``data``, which is super-resolved/downscaled 
    using the trained super-resolution network. 

    Parameters
    ----------
    model : tf.keras model
        Trained model.
    data : ndarray
        Batch of HR grids. 
    scale : int
        Scaling factor. 
    data_in_hr : bool, optional
        If True, the data is assumed to be a HR groundtruth to be downsampled. 
        Otherwise, data is a LR gridded dataset to be downscaled.
    topography : None or 2D ndarray, optional
        Elevation data.
    landocean : None or 2D ndarray, optional
        Binary land-ocean mask.
    predictors : list of ndarray, optional
        Predictor variables for trianing. Given as list of 4D ndarrays with 
        dims [nsamples, lat, lon, 1] or 5D ndarrays with dims 
        [nsamples, time, lat, lon, 1]. 
    time_window : int or None, optional
        If None, then the function assumes the ``model`` is spatial only. If an 
        integer is given, then the ``model`` should be spatio-temporal and the 
        samples are pre-processed accordingly.
    interpolation : str, optional
        Interpolation used when upsampling/downsampling the training samples.
        By default 'bicubic'. 
    save_path : str or None, optional
        If not None, the prediction (gridded variable at HR) is saved to disk.
    save_fname : str, optional
        Filename to complete the path were the prediciton is saved. 
    return_lr : bool, optional
        If True, the LR array is returned along with the downscaled one. 
    stochastic_output : bool, optional
        If True, the output will be stochastic rather than deterministic. This 
        works only when certain layers, such as dropout, are present in the 
        trained ``model``.
    """         
    timing = Timing()
    model_architecture = model.name
    if model_architecture in SPATIOTEMP_MODELS and time_window is None:
        raise ValueError('`time_window` must be provided')

    # Season is passed to each sample
    if use_season and isinstance(data, xr.DataArray):
        n_samples = data.shape[0]
        if model_architecture in SPATIOTEMP_MODELS:
            n_samples -= time_window - 1
        res = []
        lr_version = []

        for i in range(n_samples):
            if model_architecture in SPATIAL_MODELS:
                data_i = data[i]
                season = _get_season_(data_i)
                data_i = np.expand_dims(data[i], 0)
            elif model_architecture in SPATIOTEMP_MODELS:
                data_i = data[i: i+time_window]
                season = _get_season_(data_i)
                data_i = data_i.expand_dims('n_samples')
            verbose = True if i == 0 else False
            temp = _predict_(model, data_i, scale, data_in_hr, topography, 
                             landocean, predictors, season, interpolation, 
                             mean_std, save_path, save_fname, 
                             return_lr, stochastic_output, device, verbose) 
            res.append(temp[0])
            if return_lr:
                lr_version.append(temp[1])
        x_test_pred = np.array(res)
        timing.runtime()
        if return_lr:
            x_test_lr = np.array(lr_version)
            return x_test_pred, x_test_lr
        else:
            return x_test_pred
    
    # No season information is used
    elif (not use_season and isinstance(data, xr.DataArray)) or isinstance(data, np.ndarray):
        if isinstance(data, xr.DataArray):
            data = data.values
        if time_window is not None:
            data = spatial_to_temporal_samples(data, time_window)
            if predictors is not None:
                predictors = spatial_to_temporal_samples(predictors, time_window)

        res = _predict_(model, data, scale, data_in_hr, topography, landocean,
                        predictors, None, interpolation, mean_std, save_path, 
                        save_fname, return_lr, stochastic_output, device)
        timing.runtime()
        if return_lr:
            x_test_pred, x_test_lr = res
            return x_test_pred, x_test_lr
        else:
            x_test_pred = res
            return x_test_pred  


def _predict_(
    model, 
    data, 
    scale, 
    data_in_hr=True,
    topography=None, 
    landocean=None, 
    predictors=None, 
    season=None,
    interpolation='bicubic', 
    mean_std=None,
    save_path=None,
    save_fname='y_hat.npy',
    return_lr=False,
    stochastic_output=False,
    device='CPU',
    verbose=True
):
    """
    """
    model_architecture = model.name
    upsampling = model_architecture.split('_')[-1]
    if predictors is not None:
        n_predictors = len(predictors)
        predictors = np.concatenate(predictors, axis=-1)

    if model_architecture in SPATIAL_MODELS:
        if data_in_hr:
            n_samples, hr_y, hr_x, _ = data.shape
            lr_x = int(hr_x / scale)
            lr_y = int(hr_y / scale)
        else:
            n_samples, lr_y, lr_x, _ = data.shape
            hr_x = int(lr_x * scale)
            hr_y = int(lr_y * scale)
        
        n_var_channels = data.shape[-1]
        if predictors is not None:
            n_var_channels += n_predictors
            data = np.concatenate([data, predictors], axis=-1)

        aux_vars_hr = np.concatenate([checkarray_ndim(topography, 3, -1), 
                                      checkarray_ndim(landocean, 3, -1)], -1)
        aux_vars_hr = checkarray_ndim(aux_vars_hr, 4, 0)
        aux_vars_hr = np.repeat(aux_vars_hr, n_samples, axis=0)   

        if upsampling in POSTUPSAMPLING_METHODS:
            if topography is not None:
                topo_interp = resize_array(topography, (lr_x, lr_y), interpolation)
            if landocean is not None:
                # integer array can only be interpolated with nearest method
                lando_interp = resize_array(landocean, (lr_x, lr_y), interpolation='nearest')
            
            x_test_lr = np.zeros((n_samples, lr_y, lr_x, n_var_channels))  # array for inference
        
            for i in range(data.shape[0]):
                if data_in_hr:
                    # the gridded variable is downsampled
                    temparr = resize_array(data[i], (lr_x, lr_y), interpolation)
                else:
                    # the gridded variable is in LR
                    temparr = data[i]
                temparr = checkarray_ndim(temparr, ndim=3, add_axis_position=-1)
                x_test_lr[i, :, :, :] = temparr

            if topography is not None: 
                topo_interp = np.expand_dims(topo_interp, axis=0)
                topo_interp = np.repeat(topo_interp, n_samples, axis=0)      
                topo_interp = np.expand_dims(topo_interp, axis=-1)                                            
                x_test_lr = np.concatenate([x_test_lr, topo_interp], axis=-1)
            if landocean is not None:
                lando_interp = np.expand_dims(lando_interp, axis=0)  
                lando_interp = np.repeat(lando_interp, n_samples, axis=0)        
                lando_interp = np.expand_dims(lando_interp, axis=-1)   
                x_test_lr = np.concatenate([x_test_lr, lando_interp], axis=-1)
            if season is not None:
                season_array_lr = _get_season_array_(season, lr_y, lr_x)
                season_array_lr = checkarray_ndim(season_array_lr, ndim=4, add_axis_position=0)
                season_array_lr = np.repeat(season_array_lr, n_samples, axis=0)
                x_test_lr = np.concatenate([x_test_lr, season_array_lr], -1)
                season_array_hr = _get_season_array_(season, hr_y, hr_x)
                season_array_hr = checkarray_ndim(season_array_hr, ndim=4, add_axis_position=0)
                season_array_hr = np.repeat(season_array_hr, n_samples, axis=0)
                if aux_vars_hr is not None:
                    aux_vars_hr = np.concatenate([aux_vars_hr, season_array_hr], axis=-1)
                else:
                    aux_vars_hr = season_array_hr

            if verbose:
                print(f'Downsampled x_test shape: {x_test_lr.shape}')
                if aux_vars_hr is not None:
                    print(f'Aux vars shape: {aux_vars_hr.shape}')

        elif upsampling == 'pin':
            x_test_lr = np.zeros((n_samples, hr_y, hr_x, n_var_channels))

            for i in range(data.shape[0]):
                if data_in_hr:
                    x_test_resized = resize_array(data[i], (lr_x, lr_y), interpolation)  # downsampling
                else:
                    x_test_resized = data[i]  # data in LR
                # upsampling via interpolation
                x_test_resized = resize_array(x_test_resized, (hr_x, hr_y), interpolation)
                x_test_resized = checkarray_ndim(x_test_resized, ndim=3, add_axis_position=-1)
                x_test_lr[i, :, :, :] = x_test_resized
                
            if topography is not None:                                                         
                topography = np.expand_dims(topography, axis=0)
                topography = np.repeat(topography, n_samples, axis=0)      
                topography = np.expand_dims(topography, axis=-1)                                            
                x_test_lr = np.concatenate([x_test_lr, topography], axis=-1)
            if landocean is not None:
                landocean = np.expand_dims(landocean, axis=0)
                landocean = np.repeat(landocean, n_samples, axis=0)      
                landocean = np.expand_dims(landocean, axis=-1)                                            
                x_test_lr = np.concatenate([x_test_lr, landocean], axis=-1)
            if season is not None:
                season_array = _get_season_array_(season, hr_y, hr_x)
                season_array = checkarray_ndim(season_array, ndim=4, add_axis_position=0)
                season_array = np.repeat(season_array, n_samples, axis=0)
                x_test_lr = np.concatenate([x_test_lr, season_array], -1)
                if aux_vars_hr is not None:
                    aux_vars_hr = np.concatenate([aux_vars_hr, season_array], axis=-1)
                else:
                    aux_vars_hr = season_array
            
            if verbose:
                print('Downsampled x_test shape: ', x_test_lr.shape)
                if aux_vars_hr is not None:
                    print(f'Aux vars shape: {aux_vars_hr.shape}')
    
    elif model_architecture in SPATIOTEMP_MODELS:
        if predictors is not None:
            data = np.concatenate([data, predictors], axis=-1)            
        n_var_channels = data.shape[-1]

        if data_in_hr:
            n_samples, n_t, hr_y, hr_x, _ = data.shape
            lr_x = int(hr_x / scale)
            lr_y = int(hr_y / scale)
        else:
            n_samples, n_t, lr_y, lr_x, _ = data.shape

        if upsampling in POSTUPSAMPLING_METHODS:
            x_test_lr = np.zeros((n_samples, n_t, lr_y, lr_x, n_var_channels))  # array for inference
            for i in range(n_samples):
                if data_in_hr:
                    temparr = resize_array(data[i], (lr_x, lr_y), interpolation, squeezed=False)
                    temparr = checkarray_ndim(temparr, ndim=4, add_axis_position=-1)
                    x_test_lr[i] = temparr
                else:
                    x_test_lr[i] = data[i]

            if verbose:
                print('Downsampled x_test shape: ', x_test_lr.shape)

        elif upsampling == 'pin':
            x_test_lr = np.zeros((n_samples, n_t, hr_y, hr_x, n_var_channels))  # array for inference
            for i in range(n_samples):
                if data_in_hr:
                    temp = resize_array(data[i], (lr_x, lr_y), interpolation, squeezed=False)
                else:
                    temp = data[i]
                x_test_lr[i] = resize_array(temp, (hr_x, hr_y), interpolation, squeezed=False)
            
            if verbose:
                print('Downsampled x_test shape: ', x_test_lr.shape)

        if topography is not None:
            aux_vars_hr = np.expand_dims(topography, 0)
            aux_vars_hr = np.repeat(aux_vars_hr, n_samples, 0)
            aux_vars_hr = np.expand_dims(aux_vars_hr, -1)
        if landocean is not None:
            landocean = np.expand_dims(landocean, 0)
            landocean = np.repeat(landocean, n_samples, axis=0)      
            landocean = np.expand_dims(landocean, -1)
            if aux_vars_hr is not None:
                aux_vars_hr = np.concatenate([aux_vars_hr, landocean], axis=-1)
            else:
                aux_vars_hr = landocean
        if season is not None:
            season_array = _get_season_array_(season, hr_y, hr_x)
            season_array = checkarray_ndim(season_array, 4, 0)
            season_array = np.repeat(season_array, n_samples, axis=0)
            if aux_vars_hr is not None:
                aux_vars_hr = np.concatenate([aux_vars_hr, season_array], axis=-1)
            else:
                aux_vars_hr = season_array
        if verbose:
            if aux_vars_hr is not None:
                print(f'Aux vars shape: {aux_vars_hr.shape}')

    ### Casting as TF tensors, creating inputs ---------------------------------
    x_test_lr = tf.cast(x_test_lr, tf.float32)   
    lws = checkarray_ndim(np.ones((hr_y, hr_x, 2)), 4, 0)
    lws = np.repeat(lws, n_samples, axis=0)
    local_lws_array = tf.cast(lws, tf.float32)     
    if topography is not None or landocean is not None or season is not None: 
        aux_vars_hr = tf.cast(aux_vars_hr, tf.float32) 
        inputs = [x_test_lr, aux_vars_hr, local_lws_array]
    else:
        inputs = [x_test_lr, local_lws_array]
    
    ### Inference --------------------------------------------------------------
    # Stochasticity via dropout. It usually only applies when training (no values 
    # are dropped during inference). With training=True, the Dropout layer will 
    # behave in training mode and dropout will be applied at inference time
    with tf.device('/' + device + ':0'):
        x_test_pred = model(inputs, training=stochastic_output)
        x_test_pred = x_test_pred.numpy()

    if mean_std is not None:
        mean, std = mean_std
        x_test_pred *= std
        x_test_pred += mean
    
    if save_path is not None and save_fname is not None:
        name = os.path.join(save_path, save_fname)
        np.save(name, x_test_pred.astype('float32'))
    
    if return_lr:
        return x_test_pred, x_test_lr
    else:
        return x_test_pred

