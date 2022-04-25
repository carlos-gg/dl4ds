import tensorflow as tf
import numpy as np
import scipy as sc
import xarray as xr
import ecubevis as ecv

from . import POSTUPSAMPLING_METHODS
from .utils import crop_array, resize_array, checkarray_ndim


def create_pair_hr_lr(
    array, 
    array_lr,
    upsampling,
    scale, 
    patch_size, 
    static_vars=None, 
    predictors=None, 
    season=None,
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
    array_lr : np.ndarray
        LR gridded data. If not provided, then implicit/coarsened pairs are
        created from ``array``.
    upsampling : str
            String with the name of the upsampling method. 
    scale : int
        Scaling factor.
    patch_size : int or None
        Size of the square patches to be extracted, in pixels for the HR grid.
    static_vars : None or list of 2D ndarrays, optional
        Static variables such as elevation data or a binary land-ocean mask.
    predictors : np.ndarray, optional
        Predictor variables in HR. To be concatenated to the LR version of 
        `array`.
    interpolation : str, optional
        Interpolation used when upsampling/downsampling the training samples.
        By default 'bicubic'. 
    debug : bool, optional
        If True, plots and debugging information are shown.

    """
    def preproc_static_vars(var):
        if patch_size is not None:
            var_hr = crop_array(np.squeeze(var), patch_size, yx=(crop_y, crop_x))
            var_hr = checkarray_ndim(var_hr, 3, -1)
            if upsampling in POSTUPSAMPLING_METHODS:  
                var_lr = resize_array(var_hr, (patch_size_lr, patch_size_lr), interpolation) 
            else:
                var_lr = var_hr
        else:
            var_hr = checkarray_ndim(var, 3, -1)
            if upsampling in POSTUPSAMPLING_METHODS: 
                var_lr = resize_array(var, (lr_x, lr_y), interpolation)
            else:
                var_lr = var_hr 
        var_lr = checkarray_ndim(var_lr, 3, -1)

        return var_hr, var_lr
    
    # --------------------------------------------------------------------------    
    hr_array = array
    if array_lr is not None:
        lr_array = array_lr
        lr_is_given = True
    else:
        lr_is_given = False

    if hr_array.ndim == 4:
        is_spatiotemp = True
        hr_y = hr_array.shape[1]
        hr_x = hr_array.shape[2]
    elif hr_array.ndim == 3:
        is_spatiotemp = False
        hr_y = hr_array.shape[0]
        hr_x = hr_array.shape[1]

    # --------------------------------------------------------------------------
    # Cropping/resizing the arrays        
    if upsampling == 'pin': 
        if lr_is_given:
            if is_spatiotemp:
                lr_y = array_lr.shape[1]
                lr_x = array_lr.shape[2]
            else:
                lr_y = array_lr.shape[0]
                lr_x = array_lr.shape[1]
            # lr grid is upsampled via interpolation
            if is_spatiotemp:
                lr_array = checkarray_ndim(lr_array, 4, -1)
            lr_array_resized = resize_array(lr_array, (hr_x, hr_y), interpolation, squeezed=False)         
        else:
            lr_x, lr_y = int(hr_x / scale), int(hr_y / scale) 
            # hr grid is downsampled and upsampled via interpolation
            lr_array_resized = resize_array(hr_array, (lr_x, lr_y), interpolation, squeezed=False)
            # coarsened grid is upsampled via interpolation
            lr_array_resized = resize_array(lr_array_resized, (hr_x, hr_y), interpolation, squeezed=False)  
        
        if patch_size is not None:
            # cropping both hr_array and lr_array (same sizes)
            hr_array, crop_y, crop_x = crop_array(np.squeeze(hr_array), patch_size, 
                                                  yx=None, position=True)
            lr_array = crop_array(np.squeeze(lr_array_resized), patch_size, yx=(crop_y, crop_x))
        else:
            # no cropping
            lr_array = lr_array_resized
        
        if is_spatiotemp:
            hr_array = checkarray_ndim(hr_array, 4, -1)
            lr_array = checkarray_ndim(lr_array, 4, -1)
        else:
            hr_array = checkarray_ndim(hr_array, 3, -1)
            lr_array = checkarray_ndim(lr_array, 3, -1)

        if predictors is not None:
            if predictors.shape[1] != lr_y or predictors.shape[2] != lr_x:
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
            lr_array = np.concatenate([lr_array, lr_array_predictors], axis=-1)

    elif upsampling in POSTUPSAMPLING_METHODS:
        if patch_size is not None:
            patch_size_lr = int(patch_size / scale)

        if lr_is_given:
            if is_spatiotemp:
                lr_y = array_lr.shape[1]
                lr_x = array_lr.shape[2]
            else:
                lr_y = array_lr.shape[0]
                lr_x = array_lr.shape[1]            
        else:
            lr_x, lr_y = int(hr_x / scale), int(hr_y / scale)

        if predictors is not None:
            if predictors.shape[1] != lr_y or predictors.shape[2] != lr_x:
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
                lr_array = resize_array(hr_array, (lr_x, lr_y), interpolation, squeezed=False)   

            if is_spatiotemp:
                hr_array = checkarray_ndim(hr_array, 4, -1)
                lr_array = checkarray_ndim(lr_array, 4, -1)                
            else:
                hr_array = checkarray_ndim(hr_array, 3, -1)
                lr_array = checkarray_ndim(lr_array, 3, -1)

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
            hr_array = checkarray_ndim(hr_array, 3, -1)
            lr_array = checkarray_ndim(lr_array, 3, -1)

    # --------------------------------------------------------------------------
    # Including the static variables and season
    static_array_hr = []
    if static_vars is not None:
        for staticvar in static_vars:
            staticvar_hr, staticvar_lr = preproc_static_vars(staticvar)
            static_array_hr.append(staticvar_hr)
            # for spatial samples, the static array is concatenated to the lr one
            if not is_spatiotemp:
                lr_array = np.concatenate([lr_array, staticvar_lr], axis=-1)     
        static_array_hr = np.concatenate(static_array_hr, axis=-1)

    if season is not None:
        if patch_size is not None:
            season_array_hr = _get_season_array_(season, patch_size, patch_size) 
            static_array_hr = np.concatenate([static_array_hr, season_array_hr], axis=-1)
            if upsampling in POSTUPSAMPLING_METHODS:
                season_array_lr = _get_season_array_(season, patch_size_lr, patch_size_lr) 
            else:
                season_array_lr = season_array_hr
            lr_array = np.concatenate([lr_array, season_array_lr], axis=-1)
        else:
            season_array_hr = _get_season_array_(season, hr_y, hr_x) 
            static_array_hr = np.concatenate([static_array_hr, season_array_hr], axis=-1)
            if upsampling in POSTUPSAMPLING_METHODS:
                season_array_lr = _get_season_array_(season, lr_y, lr_x) 
            else:
                season_array_lr = season_array_hr

            # for spatial samples, the season array is concatenated to the lr
            if not is_spatiotemp:
                lr_array = np.concatenate([lr_array, season_array_lr], axis=-1)
    else:
        season_array_lr = None

    hr_array = np.asarray(hr_array, 'float32')
    lr_array = np.asarray(lr_array, 'float32')
    if static_vars is not None or season_array_lr is not None:
        static_array_hr = np.asanyarray(static_array_hr, 'float32')
    else:
        static_array_hr = None

    if debug: 
        if is_spatiotemp:
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

        else:
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
            
            if static_vars is not None or season is not None:
                ecv.plot_ndarray(np.moveaxis(static_array_hr, -1, 0), interactive=False, dpi=100, 
                                plot_title='HR auxiliary array')

            if predictors is not None:
                ecv.plot_ndarray(np.rollaxis(lr_array_predictors, 2, 0), dpi=100, interactive=False, 
                                plot_title='LR predictors')

    if static_vars is not None or season is not None:
        return hr_array, lr_array, static_array_hr
    else:
        return hr_array, lr_array


def create_batch_hr_lr(
    all_indices,
    index,
    array, 
    array_lr,
    upsampling,
    scale=4, 
    batch_size=32, 
    patch_size=None,
    time_window=None,
    static_vars=None, 
    predictors=None,
    interpolation='inter_area',
    time_metadata=None
    ):
    """Create a batch of HR/LR samples.
    """
    # take a batch of indices (`batch_size` indices randomized temporally)
    batch_rand_idx = all_indices[index * batch_size : (index + 1) * batch_size]
    batch_hr = []
    batch_lr = []
    batch_aux_hr = []

    # looping to create a batch of samples
    for i in batch_rand_idx:
        # spatial samples
        if time_window is None:  
            data_i = array[i]
            data_lr_i = None if array_lr is None else array_lr[i]
            predictors_i = None if predictors is None else predictors[i]
            season_i = _get_season_(time_metadata[i], time_window) if time_metadata is not None else None

        # spatio-temporal samples
        else:
            data_i = array[i:i+time_window]  
            data_lr_i = None if array_lr is None else array_lr[i:i+time_window]     
            predictors_i = None if predictors is None else predictors[i:i+time_window]
            season_i = _get_season_(time_metadata[i:i+time_window], time_window) if time_metadata is not None else None

        res = create_pair_hr_lr(
            array=data_i,
            array_lr=data_lr_i,
            upsampling=upsampling,
            scale=scale, 
            patch_size=patch_size, 
            static_vars=static_vars, 
            season=season_i,
            interpolation=interpolation,
            predictors=predictors_i)

        if static_vars is not None or season_i is not None:
            hr_array, lr_array, static_array_hr = res
            batch_aux_hr.append(static_array_hr)
        else:
            hr_array, lr_array = res
        batch_lr.append(lr_array)
        batch_hr.append(hr_array)
    batch_lr = np.asarray(batch_lr)
    batch_hr = np.asarray(batch_hr) 
    if static_vars is not None or season_i is not None:
        batch_aux_hr = np.asarray(batch_aux_hr)
        return [batch_lr, batch_aux_hr], [batch_hr]
    else:
        return [batch_lr], [batch_hr]


class DataGenerator(tf.keras.utils.Sequence):
    """
    DataGenerator creates batches of paired training samples according to the
    upsampling and other parameters. This class is used within the 
    ``dl4ds.SupervisedTrainer``.

    A sequence structure guarantees that the network will only train once on 
    each sample per epoch which is not the case with generators. 
    Every Sequence must implement the __getitem__ and the __len__ methods. If 
    you want to modify your dataset between epochs you may implement 
    on_epoch_end. The method __getitem__ should return a complete batch.

    """
    def __init__(
        self, 
        array, 
        array_lr,
        backbone,
        upsampling,
        scale, 
        batch_size=32, 
        patch_size=None,
        time_window=None,
        static_vars=None, 
        predictors=None,
        interpolation='inter_area',
        repeat=None
        ):
        """
        Parameters
        ----------
        array : np.ndarray
            HR gridded data.
        array_lr : np.ndarray
            LR gridded data. If not provided, then implicit/coarsened pairs are
            created from ``array``.
        backbone : str
            String with the name of the backbone block.
        upsampling : str
            String with the name of the upsampling method.
        scale : int
            Scaling factor.
        batch_size : int, optional
            How many samples are included in each batch. 
        patch_size : int or None
            Size of the square patches to be extracted, in pixels for the HR grid.
        time_window : int or None, optional
            If not None, then each sample will have a temporal dimension 
            (``time_window`` slices to the past are grabbed for the LR array).
        static_vars : None or list of 2D ndarrays, optional
            Static variables such as elevation data or a binary land-ocean mask.
        predictors : list of ndarray 
            List of predictor ndarrays. 
        interpolation : str, optional
            Interpolation used when upsampling/downsampling the training samples.
        repeat : int or None, optional
            Factor to repeat the samples in ``array``. Useful when ``patch_size``
            is not None.

        TO-DO
        -----
        * instead of the in-memory array, we could input the path and load the 
        netcdf files lazily or memmap a numpy array
        """        
        if isinstance(array, xr.DataArray):
            # self.time_metadata = array.time.copy()  # grabbing time metadata
            self.time_metadata = None
            self.array = array.values
        elif isinstance(array, np.ndarray):
            self.array = array
            self.time_metadata = None

        if isinstance(array_lr, xr.DataArray):
            self.array_lr = array_lr.values
        else:
            self.array_lr = array_lr
        
        self.batch_size = batch_size
        self.scale = scale
        self.upsampling = upsampling
        self.backbone = backbone
        self.patch_size = patch_size
        self.time_window = time_window
        self.static_vars = static_vars
        if self.static_vars is not None:
            for i in range(len(self.static_vars)):
                if isinstance(self.static_vars[i], xr.DataArray):
                    self.static_vars[i] = self.static_vars[i].values
        self.predictors = predictors
        # concatenating list of ndarray variables along the last dimension  
        if self.predictors is not None:
            self.predictors = np.concatenate(self.predictors, axis=-1)
        self.interpolation = interpolation
        self.repeat = repeat
        
        # shuffling the order of the available indices (n samples)
        if self.time_window is not None:
            self.n = self.array.shape[0] - self.time_window
        else:
            self.n = self.array.shape[0]
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
        res = create_batch_hr_lr(
            self.indices,
            index,
            self.array, 
            self.array_lr,
            upsampling=self.upsampling,
            scale=self.scale, 
            batch_size=self.batch_size, 
            patch_size=self.patch_size,
            time_window=self.time_window,
            static_vars=self.static_vars, 
            predictors=self.predictors,
            interpolation=self.interpolation,
            time_metadata=self.time_metadata)

        return res


def _get_season_(time_metadata, time_window):
    """ Get the season for a given sample.
    """
    if time_window is None:
        month_int = time_metadata.dt.month.values
    else:
        month_int = sc.stats.mode(time_metadata.time.dt.month.values)
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