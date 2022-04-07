from datetime import time
import os
import numpy as np
import xarray as xr
import tensorflow as tf

from .utils import Timing, checkarray_ndim, resize_array
from .dataloader import create_batch_hr_lr
from .training import CGANTrainer, SupervisedTrainer


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
        use_season=False,
        time_window=None,
        time_metadata=None,
        interpolation='inter_area', 
        batch_size=64,
        save_path=None,
        save_fname='y_hat.npy',
        return_lr=False,
        device='GPU'):
        """ 
        Parameters
        ----------
        trainer : dl4ds.SupervisedTrainer or dl4ds.CGANTrainer
            Trainer containing a keras model (``model`` or ``generator``).
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
        self.use_season = use_season 
        self.time_window = time_window
        self.time_metadata = time_metadata
        self.interpolation = interpolation 
        self.batch_size = batch_size
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
            use_season=self.use_season,
            static_vars=self.static_vars, 
            predictors=self.predictors,
            time_window=self.time_window, 
            time_metadata=self.time_metadata, 
            interpolation=self.interpolation, 
            batch_size=self.batch_size, 
            save_path=self.save_path,
            save_fname=self.save_fname, 
            return_lr=self.return_lr,
            device=self.device) 


def predict(
    trainer, 
    array, 
    scale, 
    array_in_hr=True,
    use_season=False,
    static_vars=None, 
    predictors=None, 
    time_window=None,
    time_metadata=None,
    interpolation='inter_area', 
    batch_size=64,
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
    """         
    timing = Timing()

    if isinstance(trainer, SupervisedTrainer):
        model = trainer.model
    elif isinstance(trainer, CGANTrainer):
        model = trainer.generator

    upsampling = model.name.split('_')[-1]
    # if (check input dimensionality) and time_window is None:
    #    raise ValueError('`time_window` must be provided')

    if use_season:
        if isinstance(array, xr.DataArray):
            time_metadata = array.time.copy()
        else:
            if time_metadata is None:
                msg = 'when `use_season` is True, `data` must be a xr.DataArray'
                msg += ' or `time_metadata` must be provided'
                raise ValueError(msg)
            else:
                time_metadata = time_metadata
    else:
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

    # concatenating list of ndarray variables along the last dimension  
    if predictors is not None:
        predictors = np.concatenate(predictors, axis=-1)

    # when array is in LR, it gets upsampled according to scale
    if array_in_hr:
        array_hr = array
        array_lr = None
    else:
        array = checkarray_ndim(array, 4, -1)
        hr_x = array.shape[2] * scale
        hr_y = array.shape[1] * scale
        array_hr = resize_array(array, (hr_x, hr_y), interpolation, squeezed=False) 
        array_lr = array

    batch = create_batch_hr_lr(       
        np.arange(n_samples),
        0,
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

    if static_vars is not None or use_season:
        [batch_lr, batch_aux_hr], _ = batch
    else:
        [batch_lr], _ = batch

    x_test_lr = batch_lr  

    ### Casting as TF tensors, creating inputs ---------------------------------
    x_test_lr = tf.cast(x_test_lr, tf.float32)   
    if static_vars is not None or use_season: 
        aux_vars_hr = tf.cast(batch_aux_hr, tf.float32) 
        inputs = [x_test_lr, aux_vars_hr]
    else:
        inputs = [x_test_lr]
    
    ### Inference --------------------------------------------------------------
    with tf.device('/' + device + ':0'):
        # https://www.tensorflow.org/api_docs/python/tf/keras/Model#predict
        x_test_hat = model.predict(inputs, batch_size=batch_size, verbose=1)
    
    if save_path is not None and save_fname is not None:
        name = os.path.join(save_path, save_fname)
        np.save(name, x_test_hat.astype('float32'))
    
    timing.runtime()
    if return_lr:
        return x_test_hat, np.array(x_test_lr)
    else:
        return x_test_hat        
    
