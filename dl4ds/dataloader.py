import tensorflow as tf
import numpy as np
import scipy as sc
import xarray as xr

import sys
sys.path.append('/gpfs/home/bsc32/bsc32409/src/ecubevis/')
import ecubevis as ecv

from . import POSTUPSAMPLING_METHODS
from .utils import crop_array, resize_array, checkarg_model, checkarray_ndim


def create_batch_hr_lr(x_train, batch_size, predictors, scale, topography, 
                       landocean, patch_size, time_window, model, interpolation, 
                       shuffle=True):
    """Create a batch of HR/LR samples. Used in the adversarial conditional 
    training.
    """
    batch_hr = []
    batch_lr = []
    batch_auxvars = []
    batch_lws = []

    if time_window is None:
        if shuffle:
            indices = np.random.choice(x_train.shape[0], batch_size, replace=False)
        else:
            indices = np.arange(x_train.shape[0])

        for i in indices:
            if predictors is not None:
                params = dict(predictors=predictors[i])
            else:
                params = dict()
            
            if isinstance(x_train, xr.DataArray):
                season = _get_season_(x_train[i])
            else:
                season = None

            res = create_pair_hr_lr(
                x_train[i],
                scale=scale, 
                topography=topography, 
                landocean=landocean, 
                season=season,
                patch_size=patch_size, 
                model=model, 
                interpolation=interpolation,
                **params)

            hr_array = res[0]
            lr_array = res[1]
            batch_lr.append(lr_array)
            batch_hr.append(hr_array)
            if topography is not None or landocean is not None or season is not None:
                batch_auxvars.append(res[2])
                batch_lws.append(res[3])
            else:
                batch_lws.append(res[2])
    
    else:
        if shuffle:
            rangevec = np.arange(time_window, x_train.shape[0])
            indices = np.random.choice(rangevec, batch_size, replace=False)
        else:
            indices = np.arange(time_window, x_train.shape[0])

        for i in indices:
            if predictors is not None:
                params = dict(predictors=predictors[i-time_window: i])
            else:
                params = dict()

            if isinstance(x_train, xr.DataArray):
                season = _get_season_(x_train[i-time_window: i])
            else:
                season = None

            res = create_pair_temp_hr_lr(
                x_train[i-time_window: i],
                scale=scale, 
                topography=topography, 
                landocean=landocean, 
                season=season,
                patch_size=patch_size, 
                model=model, 
                interpolation=interpolation,
                **params)

            hr_array = res[0]
            lr_array = res[1]
            batch_lr.append(lr_array)
            batch_hr.append(hr_array)
            if topography is not None or landocean is not None or season is not None:
                batch_auxvars.append(res[2])
                batch_lws.append(res[3])
            else:
                batch_lws.append(res[2])

    batch_lr = np.asarray(batch_lr)
    batch_hr = np.asarray(batch_hr) 
    batch_lws = np.asarray(batch_lws)
    if topography is not None or landocean is not None or season is not None:
        batch_auxvars = np.asarray(batch_auxvars)
        return batch_hr, batch_lr, batch_auxvars, batch_lws
    else:
        return batch_hr, batch_lr, batch_lws
 

def create_pair_hr_lr(
    array, 
    array_lr,
    scale, 
    patch_size, 
    topography=None, 
    landocean=None, 
    predictors=None, 
    season=None,
    model='resnet_spc',
    debug=False, 
    interpolation='inter_area'):
    """
    Create a pair of HR and LR square sub-patches. In this case, the LR 
    corresponds to a coarsen version of the HR reference with land-ocean mask,
    topography and auxiliary predictors added as "image channels".

    Parameters
    ----------
    array : np.ndarray
        HR gridded data.
    scale : int
        Scaling factor.
    patch_size : int or None
        Size of the square patches to be extracted, in pixels for the HR grid.
    topography : None or 2D ndarray, optional
        Elevation data.
    landocean : None or 2D ndarray, optional
        Binary land-ocean mask.
    predictors : np.ndarray, optional
        Predictor variables in HR. To be concatenated to the LR version of 
        `array`.
    model : str, optional
        String with the name of the model architecture.
    interpolation : str, optional
        Interpolation used when upsampling/downsampling the training samples.
        By default 'bicubic'. 
    debug : bool, optional
        If True, plots and debugging information are shown.

    """
    upsampling_method = model.split('_')[-1]

    if isinstance(array, xr.DataArray):
        array = array.values
    if isinstance(array_lr, xr.DataArray):
        array_lr = array_lr.values
    hr_array = np.squeeze(array)
    if array_lr is not None:
        lr_array = np.squeeze(array_lr)
        lr_is_given = True
    else:
        lr_is_given = False
    hr_y, hr_x = hr_array.shape

    # --------------------------------------------------------------------------
    # Cropping/resizing the arrays        
    if upsampling_method == 'pin': 
        if lr_is_given:
            lr_y = array_lr.shape[0]
            lr_x = array_lr.shape[1]
            # lr grid is upsampled via interpolation
            lr_array_resized = resize_array(lr_array, (hr_x, hr_y), interpolation)  
        else:
            lr_x, lr_y = int(hr_x / scale), int(hr_y / scale) 
            # hr grid is downsampled and upsampled via interpolation
            lr_array_resized = resize_array(hr_array, (lr_x, lr_y), interpolation)
            # coarsened grid is upsampled via interpolation
            lr_array_resized = resize_array(lr_array_resized, (hr_x, hr_y), interpolation)  
        
        if patch_size is not None:
            # cropping both hr_array and lr_array (same sizes)
            hr_array, crop_y, crop_x = crop_array(np.squeeze(hr_array), patch_size, 
                                                  yx=None, position=True)
            lr_array = crop_array(np.squeeze(lr_array_resized), patch_size, yx=(crop_y, crop_x))
        else:
            # no cropping
            lr_array = lr_array_resized
        hr_array = np.expand_dims(hr_array, -1)
        lr_array = np.expand_dims(lr_array, -1)
    
        if predictors is not None:
            if predictors.shape[1] != lr_y and predictors.shape[2] != lr_x:
                # we coarsen/interpolate the mid-res or high-res predictors
                predictors = resize_array(predictors, (lr_x, lr_y), interpolation)  

            predictors = resize_array(predictors, (hr_x, hr_y), interpolation)  
            if patch_size is not None:
                # cropping first the predictors 
                lr_array_predictors, crop_y, crop_x = crop_array(predictors, patch_size,
                                                                 yx=(crop_y, crop_x), position=True)
            else:
                lr_array_predictors = predictors
            # concatenating the predictors to the lr image
            lr_array = np.concatenate([lr_array, lr_array_predictors], axis=2)

    elif upsampling_method in POSTUPSAMPLING_METHODS:
        if patch_size is not None:
            patch_size_lr = int(patch_size / scale)

        if lr_is_given:
            lr_y = array_lr.shape[0]
            lr_x = array_lr.shape[1]
        else:
            lr_x, lr_y = int(hr_x / scale), int(hr_y / scale)
    
        if predictors is not None:
            if predictors.shape[1] != lr_y and predictors.shape[2] != lr_x:
                # we coarsen/interpolate the mid-res or high-res predictors
                lr_array_predictors = resize_array(predictors, (lr_x, lr_y), interpolation) 
            else:
                lr_array_predictors = predictors 

            if patch_size is not None:
                # cropping the lr predictors 
                lr_array_predictors, crop_y, crop_x = crop_array(lr_array_predictors, patch_size_lr,
                                                                 yx=None, position=True)
                crop_y_hr = int(crop_y * scale)
                crop_x_hr = int(crop_x * scale)
                # cropping the hr_array
                hr_array = crop_array(np.squeeze(hr_array), patch_size, yx=(crop_y_hr, crop_x_hr))   
                if lr_is_given:
                    lr_array = crop_array(lr_array, patch_size_lr, yx=(crop_y, crop_x))
            
            # downsampling the hr array to get lr_array when the lr array is not provided
            if not lr_is_given:
                lr_array = resize_array(hr_array, (lr_x, lr_y), interpolation)       
            hr_array = np.expand_dims(hr_array, -1)
            lr_array = np.expand_dims(lr_array, -1) 
            # concatenating the predictors to the lr grid
            lr_array = np.concatenate([lr_array, lr_array_predictors], axis=-1)
        else:
            if patch_size is not None:
                if lr_is_given:
                    # cropping the lr array
                    lr_array, crop_y, crop_x = crop_array(lr_array, patch_size_lr,
                                                          yx=None, position=True)
                    crop_y_hr = int(crop_y * scale)
                    crop_x_hr = int(crop_x * scale)
                    # cropping the hr_array
                    hr_array = crop_array(np.squeeze(hr_array), patch_size, yx=(crop_y_hr, crop_x_hr)) 
                else:
                    # cropping the hr array 
                    hr_array, crop_y, crop_x = crop_array(hr_array, patch_size, yx=None, position=True)
                    # downsampling the hr array to get lr_array
                    lr_array = resize_array(hr_array, (patch_size_lr, patch_size_lr), interpolation)
            else:
                if not lr_is_given:
                    # downsampling the hr array to get lr_array
                    lr_array = resize_array(hr_array, (lr_x, lr_y), interpolation)    
            hr_array = np.expand_dims(hr_array, -1)
            lr_array = np.expand_dims(lr_array, -1)

    # --------------------------------------------------------------------------
    # Including the static variables and season
    if topography is not None:
        if patch_size is not None:
            topo_hr = crop_array(np.squeeze(topography), patch_size, yx=(crop_y, crop_x))
            static_array_hr = checkarray_ndim(topo_hr, 3, -1)
            if upsampling_method in POSTUPSAMPLING_METHODS:  
                topo_concat = resize_array(topo_hr, (patch_size_lr, patch_size_lr), interpolation) 
            else:
                topo_concat = topo_hr
        else:
            static_array_hr = checkarray_ndim(topography, 3, -1)
            if upsampling_method in POSTUPSAMPLING_METHODS: 
                topo_concat = resize_array(topography, (lr_x, lr_y), interpolation)
            else:
                topo_concat = topography
        topo_concat = checkarray_ndim(topo_concat, 3, -1)
        lr_array = np.concatenate([lr_array, topo_concat], axis=-1)        

    if landocean is not None:
        if patch_size is not None:
            landocean_hr = crop_array(np.squeeze(landocean), patch_size, yx=(crop_y, crop_x))
            if upsampling_method in POSTUPSAMPLING_METHODS:  
                lando_concat = resize_array(landocean_hr, (patch_size_lr, patch_size_lr), 'nearest')  
            else:
                lando_concat = landocean_hr
        else:
            landocean_hr = landocean
            if upsampling_method in POSTUPSAMPLING_METHODS: 
                lando_concat = resize_array(landocean, (lr_x, lr_y), 'nearest')      
            else:
                lando_concat = landocean
        lando_concat = checkarray_ndim(lando_concat, 3, -1)
        lr_array = np.concatenate([lr_array, lando_concat], axis=-1) 
        static_array_hr = np.concatenate([static_array_hr, checkarray_ndim(landocean_hr, 3, -1)], axis=-1)

    if season is not None:
        if patch_size is not None:
            season_array_hr = _get_season_array_(season, patch_size, patch_size) 
            static_array_hr = np.concatenate([static_array_hr, season_array_hr], axis=-1)
            if upsampling_method in POSTUPSAMPLING_METHODS:
                season_array_lr = _get_season_array_(season, patch_size_lr, patch_size_lr) 
            else:
                season_array_lr = season_array_hr
            lr_array = np.concatenate([lr_array, season_array_lr], axis=-1)
        else:
            season_array_hr = _get_season_array_(season, hr_y, hr_x) 
            static_array_hr = np.concatenate([static_array_hr, season_array_hr], axis=-1)
            if upsampling_method in POSTUPSAMPLING_METHODS:
                season_array_lr = _get_season_array_(season, lr_y, lr_x) 
            else:
                season_array_lr = season_array_hr
            lr_array = np.concatenate([lr_array, season_array_lr], axis=-1)

    hr_array = np.asarray(hr_array, 'float32')
    lr_array = np.asarray(lr_array, 'float32')
    if topography is not None or landocean is not None or season is not None:
        static_array_hr = np.asanyarray(static_array_hr, 'float32')
    # Including the lws array --------------------------------------------------
    local_lws_array = np.ones((hr_y, hr_x, 2))
    local_lws_array = np.asarray(local_lws_array, 'float32')

    if debug: 
        if static_array_hr is not None:
            print(f'HR array: {hr_array.shape}, LR array {lr_array.shape}, Auxiliary array HR {static_array_hr.shape}')
        else:
            print(f'HR array: {hr_array.shape}, LR array {lr_array.shape}')
        if patch_size is not None:
            print(f'Crop X,Y: {crop_x}, {crop_y}')
        
        ecv.plot_ndarray(np.squeeze(hr_array), dpi=100, interactive=False, 
                         subplot_titles='HR array')
        
        ecv.plot_ndarray(np.moveaxis(np.squeeze(lr_array), -1, 0), dpi=100, interactive=False, 
                         plot_title='LR array')
        
        if topography is not None or landocean is not None or season is not None:
            ecv.plot_ndarray(np.moveaxis(static_array_hr, -1, 0), interactive=False, dpi=100, 
                             plot_title='HR auxiliary array')

        if predictors is not None:
            ecv.plot_ndarray(np.rollaxis(lr_array_predictors, 2, 0), dpi=100, interactive=False, 
                             plot_title=('LR predictors'))

    if topography is not None or landocean is not None or season is not None:
        return hr_array, lr_array, static_array_hr, local_lws_array
    else:
        return hr_array, lr_array, local_lws_array


def create_pair_temp_hr_lr(
    array, 
    scale, 
    patch_size, 
    topography=None, 
    landocean=None, 
    predictors=None, 
    season=None,
    model='resnet_spc',
    debug=False, 
    interpolation='inter_area'):
    """
    Create a pair of HR and LR square sub-patches with a temporal window. In 
    this case, the LR corresponds to a coarsen version of the HR reference. If
    a land-ocean mask or topography are provided, these are concatenated and 
    returned as an additional output.

    Parameters
    ----------
    array : np.ndarray
        HR gridded data.
    scale : int
        Scaling factor.
    patch_size : int or None
        Size of the square patches to be extracted.
    topography : None or 2D ndarray, optional
        Elevation data.
    landocean : None or 2D ndarray, optional
        Binary land-ocean mask.
    predictors : np.ndarray, optional
        Predictor variables in HR. To be concatenated to the LR version of 
        `array`.
    model : str, optional
        String with the name of the model architecture.
    interpolation : str, optional
        Interpolation used when upsampling/downsampling the training samples.
        By default 'bicubic'. 
    debug : bool, optional
        If True, plots and debugging information are shown.
    
    """
    if isinstance(array, xr.DataArray):
        array = array.values
    hr_y = array.shape[1] 
    hr_x = array.shape[2]
    hr_array = array  # 4D array [time,y,x,1ch]
    upsampling_method = model.split('_')[-1]

    # --------------------------------------------------------------------------
    # Cropping/resizing the arrays
    if upsampling_method == 'pin': 
        lr_x, lr_y = int(hr_x / scale), int(hr_y / scale) 
        # HR array is downsampled and upsampled via interpolation
        lr_array_resized = resize_array(hr_array, (lr_x, lr_y), interpolation, squeezed=False)
        lr_array_resized = resize_array(lr_array_resized, (hr_x, hr_y), interpolation, squeezed=False)
        lr_array = lr_array_resized
        if patch_size is not None:
            # cropping both hr_array and lr_array (same sizes)
            hr_array, crop_y, crop_x = crop_array(hr_array, patch_size, yx=None, position=True)
            lr_array = crop_array(lr_array, patch_size, yx=(crop_y, crop_x))
            if predictors is not None:
                predictors = crop_array(predictors, patch_size, yx=(crop_y, crop_x), position=False)
    
    elif upsampling_method in POSTUPSAMPLING_METHODS:
        if patch_size is not None:
            lr_x, lr_y = int(patch_size / scale), int(patch_size / scale) 
            hr_array, crop_y, crop_x = crop_array(hr_array, patch_size, yx=None, position=True)
            lr_array = resize_array(hr_array, (lr_x, lr_y), interpolation, squeezed=False)
            if predictors is not None:
                predictors = crop_array(predictors, patch_size, yx=(crop_y, crop_x), position=False)
        else:
            lr_x, lr_y = int(hr_x / scale), int(hr_y / scale)
            lr_array = resize_array(hr_array, (lr_x, lr_y), interpolation, squeezed=False)
        
        # downsampling the predictors and concatenating
        if predictors is not None:
            predictors = resize_array(predictors, (lr_x, lr_y), interpolation, squeezed=False)

    if predictors is not None:    
        lr_array = np.concatenate([lr_array, predictors], axis=-1)

    # --------------------------------------------------------------------------
    # Including the static variables and season
    if topography is not None:
        if patch_size is not None:
            topography = crop_array(topography, patch_size, yx=(crop_y, crop_x))
        static_array_hr = np.expand_dims(topography, -1)

    if landocean is not None:
        if patch_size is not None:
            landocean = crop_array(landocean, patch_size, yx=(crop_y, crop_x))  
        landocean = np.expand_dims(landocean, -1)
        if static_array_hr is not None:
            static_array_hr = np.concatenate([static_array_hr, landocean], axis=-1)
        else:
            static_array_hr = landocean

    if season is not None:
        if patch_size is not None:
            season_array_hr = _get_season_array_(season, patch_size, patch_size) 
            static_array_hr = np.concatenate([static_array_hr, season_array_hr], axis=-1)
        else:
            season_array_hr = _get_season_array_(season, hr_y, hr_x) 
            static_array_hr = np.concatenate([static_array_hr, season_array_hr], axis=-1)

    hr_array = np.asarray(hr_array[:,:,:,0], 'float32')  # keeping the target variable
    hr_array = np.expand_dims(hr_array, -1)
    lr_array = np.asarray(lr_array, 'float32')
    if static_array_hr is not None:
        static_array_hr = np.asarray(static_array_hr, 'float32')
    # Including the lws array --------------------------------------------------
    local_lws_array = np.ones((hr_y, hr_x, 2))
    local_lws_array = np.asarray(local_lws_array, 'float32')

    if debug:
        print(f'HR array: {hr_array.shape}, LR array: {lr_array.shape}, Auxiliary array: {season_array_hr.shape}')
        if patch_size is not None:
            print(f'Crop X,Y: {crop_x}, {crop_y}')
    
        ecv.plot_ndarray(np.squeeze(hr_array), dpi=100, interactive=False, plot_title=('HR array'))
        for i in range(lr_array.shape[-1]):
            ecv.plot_ndarray(np.squeeze(lr_array[:,:,:,i]), dpi=100, interactive=False, 
                             plot_title=(f'LR array, variable {i+1}'))
        
        if static_array_hr is not None:
            ecv.plot_ndarray(tuple(np.moveaxis(static_array_hr, -1, 0)), interactive=False, 
                             dpi=100,plot_title='Auxiliary array HR')

    if static_array_hr is not None:
        return hr_array, lr_array, static_array_hr, local_lws_array
    else:
        return hr_array, lr_array, local_lws_array


class DataGenerator(tf.keras.utils.Sequence):
    """
    A sequence structure guarantees that the network will only train once on 
    each sample per epoch which is not the case with generators. 
    Every Sequence must implement the __getitem__ and the __len__ methods. If 
    you want to modify your dataset between epochs you may implement 
    on_epoch_end. The method __getitem__ should return a complete batch.

    """
    def __init__(self, 
        array, 
        array_lr,
        scale=4, 
        batch_size=32, 
        patch_size=None,
        time_window=None,
        topography=None, 
        landocean=None, 
        predictors=None,
        model='resnet_spc', 
        interpolation='inter_area',
        repeat=None,
        ):
        """
        Parameters
        ----------
        model : 
            Name of the model architecture. eg, 'resnet_spc', 'convnet_pin'
        predictors : list of ndarray 
            List of predictor ndarrays.

        TO-DO
        -----
        * instead of the in-memory array, we could input the path and load the 
        netcdf files lazily or memmap a numpy array
        """
        self.array = array
        self.array_lr = array_lr
        self.batch_size = batch_size
        self.scale = scale
        self.patch_size = patch_size
        self.time_window = time_window
        self.topography = topography
        self.landocean = landocean
        self.predictors = predictors
        self.model = checkarg_model(model)
        self.upsampling = self.model.split('_')[-1]
        self.interpolation = interpolation
        self.repeat = repeat
        self.n = array.shape[0]
        if self.time_window is not None:
            self.indices = np.random.permutation(np.arange(0, self.n - self.time_window))
        else:
            self.indices = np.random.permutation(np.arange(self.n))
        
        if self.repeat is not None and isinstance(self.repeat, int):
            self.indices = np.hstack([self.indices for i in range(self.repeat)])

        if patch_size is not None:
            if self.upsampling in POSTUPSAMPLING_METHODS: 
                if not self.patch_size % self.scale == 0:   
                    raise ValueError('`patch_size` must be divisible by `scale`')

    def __len__(self):
        """
        Defines the number of batches the DataGenerator can produce per epoch.
        A common practice is to set this value to n_samples / batch_size so that 
        the model sees the training samples at most once per epoch. 
        """
        n_batches = self.n // self.batch_size
        if self.repeat:
            return n_batches * self.repeat
        else:
            return n_batches

    def __getitem__(self, index):
        """
        Generate one batch of data as (X, y) value pairs where X represents the 
        input and y represents the output.
        """
        self.batch_rand_idx = self.indices[
            index * self.batch_size : (index + 1) * self.batch_size]
        batch_hr = []
        batch_lr = []
        batch_aux_hr = []
        batch_lws = []
        params = {}

        if self.predictors is not None:
            array_predictors = np.concatenate(self.predictors, axis=-1)

        for i in self.batch_rand_idx:

            if isinstance(self.array, xr.DataArray):
                season = _get_season_(self.array[i])
            else:
                season = None

            # looping to create a batch of spatial samples
            if self.time_window is None:  
                # concatenating list of ndarray variables along the last 
                # dimension to create a single ndarray 
                if self.predictors is not None:
                    params = dict(predictors=array_predictors[i])

                res = create_pair_hr_lr(
                    array=self.array[i],
                    array_lr=None if self.array_lr is None else self.array_lr[i],
                    scale=self.scale, 
                    patch_size=self.patch_size, 
                    topography=self.topography, 
                    season=season,
                    landocean=self.landocean, 
                    model=self.model,
                    interpolation=self.interpolation,
                    **params)
     
            # looping to create a batch of spatio-temporal samples
            else:
                # concatenating list of ndarray variables along the last 
                # dimension to create a single ndarray 
                if self.predictors is not None:
                    params = dict(predictors=array_predictors[i:i+self.time_window])                    

                res = create_pair_temp_hr_lr(
                    array=self.array[i:i+self.time_window],
                    scale=self.scale, 
                    patch_size=self.patch_size, 
                    topography=self.topography, 
                    landocean=self.landocean, 
                    season=season,
                    model=self.model,
                    interpolation=self.interpolation,
                    **params)

            if self.topography is not None or self.landocean is not None or season is not None:
                hr_array, lr_array, static_array_hr, lws = res
                batch_aux_hr.append(static_array_hr)
            else:
                hr_array, lr_array, lws = res
            batch_lr.append(lr_array)
            batch_hr.append(hr_array)
            batch_lws.append(lws)
        batch_lr = np.asarray(batch_lr)
        batch_hr = np.asarray(batch_hr) 
        batch_lws = np.asarray(batch_lws)
        if self.topography is not None or self.landocean is not None or season is not None:
            batch_aux_hr = np.asarray(batch_aux_hr)
            return [batch_lr, batch_aux_hr, batch_lws], [batch_hr]
        else:
            return [batch_lr, batch_lws], [batch_hr]

    def on_epoch_end(self):
        """
        """
        np.random.shuffle(self.indices)


def _get_season_(dataarray):
    """ Get the season for a single time step xr.DataArray.
    """
    if dataarray.ndim == 3:  # [lat, lon, var]
        month_int = dataarray.time.dt.month.values
    elif dataarray.ndim == 4:  # [time, lat, lon, var]
        month_int = sc.stats.mode(dataarray.time.dt.month.values)
        month_int = int(month_int.count)

    if month_int in [12, 1, 2]:
        season = 'winter'
    elif month_int in [3, 4, 5]:
        season = 'spring'
    elif month_int in [6, 7, 8]: 
        season = 'summer'
    elif month_int in [9, 10, 11]:
        season = 'autumn'
    return season


def _get_season_array_(season, sizey, sizex):
    """ Produce a multichannel array encoding the season. 
    """
    if season not in ['winter', 'spring', 'summer', 'autumn']:
        raise ValueError('``season`` not recognized')
    season_array = np.zeros((sizey, sizex, 4))
    if season == 'winter':
        season_array[:,:,0] += 1
    elif season == 'spring':
        season_array[:,:,1] += 1
    elif season == 'summer':
        season_array[:,:,2] += 1
    elif season == 'autumn':
        season_array[:,:,3] += 1    
    return season_array