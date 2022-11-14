from datetime import time
import os
import numpy as np
import xarray as xr
import tensorflow as tf
import keras

from .utils import Timing, checkarray_ndim, resize_array, spatiotemporal_to_spatial_samples
from .dataloader import create_batch_hr_lr


class Predictor():
    """     
    Predictor class for performing inference on unseen HR or LR data. The data 
    (``array``) is super-resolved or downscaled using the trained 
    super-resolution network (contained in ``trainer``).    
    """
    def __init__(
        self,
        trainer, 
        array,
        scale, 
        array_in_hr=False,
        static_vars=None,
        predictors=None,
        time_window=None,
        time_metadata=None,
        interpolation='inter_area', 
        batch_size=64,
        scaler=None,
        save_path=None,
        save_fname='y_hat.npy',
        return_lr=False,
        device='GPU'):
        """ 
        Parameters
        ----------
        trainer : dl4ds.SupervisedTrainer or dl4ds.CGANTrainer
            Trainer containing a keras model (``model`` or ``generator``). 
            Optionally, you can direclty pass the tf.keras model.
        array : ndarray
            Batch of HR grids. 
        scale : int
            Scaling factor. 
        array_in_hr : bool, optional
            If True, the data is assumed to be a HR groundtruth to be downsampled. 
            Otherwise, data is a LR gridded dataset to be downscaled.
        static_vars : None or list of 2D ndarrays, optional
                Static variables such as elevation data or binary masks.
        predictors : list of ndarray, optional
            Predictor variables for trianing. Given as list of 4D ndarrays with 
            dims [nsamples, lat, lon, 1] or 5D ndarrays with dims 
            [nsamples, time, lat, lon, 1]. 
        time_window : int or None, optional
            If None, then the function assumes the ``model`` is spatial only. If 
            an integer is given, then the ``model`` should be spatio-temporal 
            and the samples are pre-processed accordingly.
        interpolation : str, optional
            Interpolation used when upsampling/downsampling the training samples.
            By default 'bicubic'. 
        batch_size : int, optional
            Batch size for feeding samples for inference.
        scaler : None or dl4ds scaler object, optional
            Scaler for backward scaling and restoring original distribution.
        save_path : str or None, optional
            If not None, the prediction (gridded variable at HR) is saved to disk.
        save_fname : str, optional
            Filename to complete the path were the prediciton is saved.     
        return_lr : bool, optional
            If True, the LR array is returned along with the downscaled one.                                                                
        """
        self.trainer = trainer 
        self.array_in_hr = array_in_hr
        self.array = array
        self.scale = scale
        self.static_vars = static_vars
        self.predictors = predictors
        self.time_window = time_window
        self.time_metadata = time_metadata
        self.interpolation = interpolation 
        self.batch_size = batch_size
        self.scaler = scaler
        self.save_path = save_path
        self.save_fname = save_fname
        self.return_lr = return_lr
        self.device = device

    def run(self): 
        """ 
        """
        return predict(
            trainer=self.trainer, 
            array=self.array, 
            scale=self.scale, 
            array_in_hr=self.array_in_hr, 
            static_vars=self.static_vars, 
            predictors=self.predictors,
            time_window=self.time_window, 
            time_metadata=self.time_metadata, 
            interpolation=self.interpolation, 
            batch_size=self.batch_size, 
            scaler=self.scaler,
            save_path=self.save_path,
            save_fname=self.save_fname, 
            return_lr=self.return_lr,
            device=self.device) 


def predict(
    trainer, 
    array, 
    scale, 
    array_in_hr=True,
    static_vars=None, 
    predictors=None, 
    time_window=None,
    time_metadata=None,
    interpolation='inter_area', 
    batch_size=64,
    scaler=None,
    save_path=None,
    save_fname='y_hat.npy',
    return_lr=False,
    device='GPU'):
    """Inference on unseen HR or LR data. The data (``array``) is super-resolved 
    or downscaled using the trained super-resolution network (``model``). 

    Parameters
    ----------
    trainer : dl4ds.SupervisedTrainer or dl4ds.CGANTrainer
        Trainer containing a keras model (``model`` or ``generator``). 
        Optionally, you can direclty pass the tf.keras model.
    array : ndarray
        Batch of HR grids. 
    scale : int
        Scaling factor. 
    array_in_hr : bool, optional
        If True, the data is assumed to be a HR groundtruth to be downsampled. 
        Otherwise, data is a LR gridded dataset to be downscaled.
    static_vars : None or list of 2D ndarrays, optional
        Static variables such as elevation data or a binary land-ocean mask.
    predictors : list of ndarray, optional
        Predictor variables for trianing. Given as list of 4D ndarrays with dims 
        [nsamples, lat, lon, 1] or 5D ndarrays with dims [nsamples, time, lat, lon, 1]. 
    time_window : int or None, optional
        If None, then the function assumes the ``model`` is spatial only. If an 
        integer is given, then the ``model`` should be spatio-temporal and the 
        samples are pre-processed accordingly.
    interpolation : str, optional
        Interpolation used when upsampling/downsampling the training samples.
        By default 'bicubic'. 
    batch_size : int, optional
        Batch size for feeding samples for inference.
    scaler : None or dl4ds scaler object, optional
        Scaler for backward scaling and restoring original distribution.
    save_path : str or None, optional
        If not None, the prediction (gridded variable at HR) is saved to disk.
    save_fname : str, optional
        Filename to complete the path were the prediciton is saved. 
    return_lr : bool, optional
        If True, the LR array is returned along with the downscaled one. 
    """         
    timing = Timing()

    if hasattr(trainer, 'model'):
        model = trainer.model
    elif hasattr(trainer, 'generator'):
        model = trainer.generator
    else:
        model = trainer

    upsampling = model.name.split('_')[-1]
    dim = len(model.input.shape)
    if dim == 5 and time_window is None:
       raise ValueError('`time_window` must be provided for spatiotemporal model')

    time_metadata = None

    if isinstance(array, xr.DataArray):    
        array = array.values  

    if static_vars is not None:
        for i in range(len(static_vars)):
            if isinstance(static_vars[i], xr.DataArray):
                static_vars[i] = static_vars[i].values

    n_samples = array.shape[0]
    if time_window is not None:
        n_samples -= time_window - 1

    ### Concatenating list of ndarray variables along the last dimension  
    if predictors is not None:
        predictors = np.concatenate(predictors, axis=-1)

    ### Array is upsampled according to scale when array is in LR
    if array_in_hr:
        array_hr = array
        array_lr = None
    else:
        array = checkarray_ndim(array, 4, -1)
        hr_xy = (array.shape[2] * scale, array.shape[1] * scale)
        array_hr = resize_array(array, hr_xy, interpolation, squeezed=False) 
        array_lr = array

    batch = create_batch_hr_lr(       
        all_indices=np.arange(n_samples),
        index=0,
        array=array_hr, 
        array_lr=array_lr,
        upsampling=upsampling,
        scale=scale, 
        batch_size=n_samples, 
        patch_size=None,
        time_window=time_window,
        static_vars=static_vars, 
        predictors=predictors,
        interpolation=interpolation,
        time_metadata=time_metadata)

    if static_vars is not None:
        [batch_lr, batch_aux_hr], _ = batch
    else:
        [batch_lr], _ = batch

    x_test_lr = batch_lr  

    ### Casting as TF tensors, creating inputs ---------------------------------
    x_test_lr = tf.cast(x_test_lr, tf.float32)   
    if static_vars is not None: 
        aux_vars_hr = tf.cast(batch_aux_hr, tf.float32) 
        inputs = [x_test_lr, aux_vars_hr]
    else:
        inputs = [x_test_lr]
    
    ### Inference --------------------------------------------------------------
    # https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
    with tf.device('/' + device + ':0'):
        out = model.predict(inputs, batch_size=batch_size, verbose=1)
    
    ### 
    if out.ndim == 5 and time_window is not None:
        out = spatiotemporal_to_spatial_samples(out, time_window)

    if scaler is not None:
        out = scaler.inverse_transform(out)

    if save_path is not None and save_fname is not None:
        name = os.path.join(save_path, save_fname)
        np.save(name, out.astype('float32'))
    
    timing.runtime()
    if return_lr:
        return out, np.array(x_test_lr)
    else:
        return out        
    
