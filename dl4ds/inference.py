import os
import numpy as np
import xarray as xr
import tensorflow as tf

from .utils import Timing
from . import SPATIAL_MODELS, SPATIOTEMP_MODELS, POSTUPSAMPLING_METHODS
from .dataloader import _get_season_, create_pair_hr_lr, create_pair_temp_hr_lr
from .training_logic import CGANTrainer, SupervisedTrainer


def predict(
    model, 
    array, 
    scale, 
    data_in_hr=True,
    use_season=True,
    topography=None, 
    landocean=None, 
    predictors=None, 
    time_window=None,
    interpolation='inter_area', 
    save_path=None,
    save_fname='y_hat.npy',
    return_lr=False,
    stochastic_output=False,
    device='GPU'):
    """Inference with ``model`` on ``array``, which is super-resolved/downscaled 
    using the trained super-resolution network. 

    Parameters
    ----------
    model : tf.keras model
        Trained model.
    array : ndarray
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

    if isinstance(model, SupervisedTrainer):
        model = model.model
    elif isinstance(model, CGANTrainer):
        model = model.generator

    model_architecture = model.name
    if model_architecture in SPATIOTEMP_MODELS and time_window is None:
        raise ValueError('`time_window` must be provided')

    # Season is passed to each sample
    if use_season and not isinstance(array, xr.DataArray):
        raise ValueError('when `use_season` is True, `data` must be a xr.DataArray')
 
    n_samples = array.shape[0]
    if model_architecture in SPATIOTEMP_MODELS:
        n_samples -= time_window - 1
    batch_hr = []
    batch_lr = []
    batch_aux_hr = []
    batch_lws = []
    params = {}

    if predictors is not None:
        array_predictors = np.concatenate(predictors, axis=-1)

    for i in range(n_samples):

        if isinstance(array, xr.DataArray):
            season = _get_season_(array[i])
        else:
            season = None

        if model_architecture in SPATIAL_MODELS:
            # concatenating list of ndarray variables along the last 
            # dimension to create a single ndarray 
            if predictors is not None:
                params = dict(predictors=array_predictors[i])                

            dataloader_res = create_pair_hr_lr(
                array=array[i],
                scale=scale, 
                patch_size=None, 
                topography=topography, 
                season=season,
                landocean=landocean, 
                model=model_architecture,
                interpolation=interpolation,
                **params)

        elif model_architecture in SPATIOTEMP_MODELS:
            # concatenating list of ndarray variables along the last 
            # dimension to create a single ndarray 
            if predictors is not None:
                params = dict(predictors=array_predictors[i:i+time_window])

            dataloader_res = create_pair_temp_hr_lr(
                array=array[i:i+time_window],
                scale=scale, 
                patch_size=None, 
                topography=topography, 
                landocean=landocean, 
                season=season,
                model=model_architecture,
                interpolation=interpolation,
                **params)
        
        if topography is not None or landocean is not None or season is not None:
            hr_array, lr_array, static_array_hr, lws = dataloader_res
            batch_aux_hr.append(static_array_hr)
        else:
            hr_array, lr_array, lws = dataloader_res
        batch_lr.append(lr_array)
        batch_hr.append(hr_array)
        batch_lws.append(lws)

    if data_in_hr:
        x_test_lr = batch_lr
    else:
        x_test_lr = batch_hr

    ### Casting as TF tensors, creating inputs ---------------------------------
    x_test_lr = tf.cast(x_test_lr, tf.float32)   
    local_lws_array = tf.cast(batch_lws, tf.float32)     
    if topography is not None or landocean is not None or season is not None: 
        aux_vars_hr = tf.cast(batch_aux_hr, tf.float32) 
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
    
    if save_path is not None and save_fname is not None:
        name = os.path.join(save_path, save_fname)
        np.save(name, x_test_pred.astype('float32'))
    
    timing.runtime()
    if return_lr:
        x_test_lr = np.array(x_test_lr)
        return x_test_pred,  np.moveaxis(np.squeeze(x_test_lr), -1, 1)
    else:
        return x_test_pred        
    
