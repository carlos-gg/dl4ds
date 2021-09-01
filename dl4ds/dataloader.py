import tensorflow as tf
import numpy as np

import sys
sys.path.append('/gpfs/home/bsc32/bsc32409/src/ecubevis/')
import ecubevis as ecv

from . import POSTUPSAMPLING_METHODS
from .utils import crop_array, resize_array, checkarg_model


def create_pair_temp_hr_lr(
    array, 
    scale, 
    patch_size, 
    topography=None, 
    landocean=None, 
    predictors=None, 
    model='resnet_spc',
    debug=False, 
    interpolation='bicubic'):
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
    static_array = None
    hr_y = array.shape[1] 
    hr_x = array.shape[2]
    hr_array = array  # 4D array [time,y,x,1ch]
    upsampling_method = model.split('_')[-1]

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

    if topography is not None:
        if patch_size is not None:
            topography = crop_array(topography, patch_size, yx=(crop_y, crop_x))
        static_array = np.expand_dims(topography, -1)

    if landocean is not None:
        if patch_size is not None:
            landocean = crop_array(landocean, patch_size, yx=(crop_y, crop_x))  
        landocean = np.expand_dims(landocean, -1)
        if static_array is not None:
            static_array = np.concatenate([static_array, landocean], axis=-1)
        else:
            static_array = landocean
    
    hr_array = np.asarray(hr_array[:,:,:,0], 'float32')  # keeping the target variable
    hr_array = np.expand_dims(hr_array, -1)
    lr_array = np.asarray(lr_array, 'float32')
    if static_array is not None:
        static_array = np.asarray(static_array, 'float32')

    if debug:
        print(f'HR array: {hr_array.shape}, LR array: {lr_array.shape}, Static array: {static_array.shape}')
        if patch_size is not None:
            print(f'Crop X,Y: {crop_x}, {crop_y}')
    
        ecv.plot_ndarray(np.squeeze(hr_array), dpi=80, interactive=False, plot_title=('HR array'))
        for i in range(lr_array.shape[-1]):
            ecv.plot_ndarray(np.squeeze(lr_array[:,:,:,i]), dpi=80, interactive=False, 
                             plot_title=(f'LR array, variable {i+1}'))
        
        if upsampling_method in POSTUPSAMPLING_METHODS:
            if topography is not None:
                ecv.plot_ndarray(topography, interactive=False, dpi=80, subplot_titles=('HR Topography'))
            if landocean is not None:
                ecv.plot_ndarray(landocean, interactive=False, dpi=80, subplot_titles=('HR Land Ocean mask'))
        elif upsampling_method == 'pin':
            if static_array is not None:
                if topography is not None and landocean is not None:
                    subpti = ('HR Topography', 'HR Land-Ocean mask')
                else:
                    subpti = None
                ecv.plot_ndarray(tuple(np.moveaxis(static_array, -1, 0)), interactive=False, 
                                 dpi=80, subplot_titles=subpti)

    if static_array is not None:
        return hr_array, lr_array, static_array
    else:
        return hr_array, lr_array


def create_pair_hr_lr(
    array, 
    scale, 
    patch_size, 
    topography=None, 
    landocean=None, 
    predictors=None, 
    model='resnet_spc',
    debug=False, 
    interpolation='bicubic'):
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
    hr_array = np.squeeze(array)
    hr_y, hr_x = hr_array.shape
    upsampling_method = model.split('_')[-1]

    if upsampling_method == 'pin': 
        lr_x, lr_y = int(hr_x / scale), int(hr_y / scale) 
        # whole image is downsampled and upsampled via interpolation
        lr_array_resized = resize_array(hr_array, (lr_x, lr_y), interpolation)
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
    elif upsampling_method in POSTUPSAMPLING_METHODS:
        if patch_size is not None:
            lr_x, lr_y = int(patch_size / scale), int(patch_size / scale) 
        else:
            lr_x, lr_y = int(hr_x / scale), int(hr_y / scale)

    if upsampling_method == 'pin':
        if predictors is not None:
            predictors = resize_array(predictors, (lr_x, lr_y), interpolation)
            predictors = resize_array(predictors, (hr_x, hr_y), interpolation)  # upsampling the lr predictorsz
            if patch_size is not None:
                # cropping first the predictors 
                lr_array_predictors, crop_y, crop_x = crop_array(predictors, patch_size,
                                                                 yx=(crop_y, crop_x), position=True)
            else:
                lr_array_predictors = predictors
                lr_array_predictors = np.expand_dims(lr_array_predictors, -1)
            # concatenating the predictors to the lr image
            lr_array = np.concatenate([lr_array, lr_array_predictors], axis=2)
    elif upsampling_method in POSTUPSAMPLING_METHODS:
        if predictors is not None:
            if patch_size is not None:
                # cropping first the predictors 
                lr_array_predictors, crop_y, crop_x = crop_array(predictors, lr_x,
                                                                 yx=None, position=True)
                crop_y = int(crop_y * scale)
                crop_x = int(crop_x * scale)
                hr_array = crop_array(np.squeeze(hr_array), patch_size, yx=(crop_y, crop_x))   
            else:
                lr_array_predictors = resize_array(predictors, (lr_x, lr_y), interpolation) 
                lr_array_predictors = np.expand_dims(lr_array_predictors, -1)
            lr_array = resize_array(hr_array, (lr_x, lr_y), interpolation)       
            hr_array = np.expand_dims(hr_array, -1)
            lr_array = np.expand_dims(lr_array, -1) 
            # concatenating the predictors to the lr image
            lr_array = np.concatenate([lr_array, lr_array_predictors], axis=2)
        else:
            if patch_size is not None:
                # cropping the hr array
                hr_array, crop_y, crop_x = crop_array(hr_array, patch_size, yx=None, position=True)
            # downsampling the hr array to get lr_array
            lr_array = resize_array(hr_array, (lr_x, lr_y), interpolation)    
            hr_array = np.expand_dims(hr_array, -1)
            lr_array = np.expand_dims(lr_array, -1)

    if topography is not None:
        if patch_size is not None:
            topo_hr = crop_array(np.squeeze(topography), patch_size, yx=(crop_y, crop_x))
        else:
            topo_hr = topography
        if upsampling_method in POSTUPSAMPLING_METHODS:  # downsizing the topography
            topo_lr = resize_array(topo_hr, (lr_x, lr_y), interpolation)
            lr_array = np.concatenate([lr_array, np.expand_dims(topo_lr, -1)], axis=2)
        elif upsampling_method == 'pin':  # topography in HR 
            lr_array = np.concatenate([lr_array, np.expand_dims(topo_hr, -1)], axis=2)

    if landocean is not None:
        if patch_size is not None:
            landocean_hr = crop_array(np.squeeze(landocean), patch_size, yx=(crop_y, crop_x))
        else:
            landocean_hr = landocean
        if upsampling_method in POSTUPSAMPLING_METHODS:  # downsizing the land-ocean mask
            # integer array can only be interpolated with nearest method
            landocean_lr = resize_array(landocean_hr, (lr_x, lr_y), interpolation='nearest')
            lr_array = np.concatenate([lr_array, np.expand_dims(landocean_lr, -1)], axis=2)
        elif upsampling_method == 'pin':  # lando in HR 
            lr_array = np.concatenate([lr_array, np.expand_dims(landocean_hr, -1)], axis=2)
    
    hr_array = np.asarray(hr_array, 'float32')
    lr_array = np.asarray(lr_array, 'float32')

    if debug:
        print(f'HR array: {hr_array.shape}, LR array {lr_array.shape}')
        if patch_size is not None:
            print(f'Crop X,Y: {crop_x}, {crop_y}')
            ecv.plot_ndarray((array[:,:,0]), dpi=60, interactive=False)
        
        if topography is not None or landocean is not None or predictors is not None:
            lr_array_plot = np.squeeze(lr_array)[:,:,0]
        else:
            lr_array_plot = np.squeeze(lr_array)
        ecv.plot_ndarray((np.squeeze(hr_array), lr_array_plot), dpi=80, interactive=False, 
                         subplot_titles=('HR array', 'LR array'))
        
        if upsampling_method in POSTUPSAMPLING_METHODS:
            if topography is not None:
                ecv.plot_ndarray((topo_hr, topo_lr), 
                                interactive=False, dpi=80, 
                                subplot_titles=('HR Topography', 'LR Topography'))
            if landocean is not None:
                ecv.plot_ndarray((landocean_hr, landocean_lr), 
                                interactive=False, dpi=80, 
                                subplot_titles=('HR Land Ocean mask', 'LR  Land Ocean mask'))
        elif upsampling_method == 'pin':
            if topography is not None:
                ecv.plot_ndarray(topography, interactive=False, dpi=80, 
                                 subplot_titles=('HR Topography'))
            if landocean is not None:
                ecv.plot_ndarray(landocean, interactive=False, dpi=80, 
                                 subplot_titles=('HR Land Ocean mask'))

        if predictors is not None:
            ecv.plot_ndarray(np.rollaxis(lr_array_predictors, 2, 0), dpi=80, interactive=False, 
                             subplot_titles=('LR cropped predictors'), multichannel4d=True)

    return hr_array, lr_array


def create_batch_hr_lr(x_train, batch_size, predictors, scale, topography, 
                       landocean, patch_size, time_window, model, interpolation, 
                       shuffle=True):
    """Create a batch of HR/LR samples. Used in the adversarial conditional 
    training.
    """
    if time_window is None:
        if shuffle:
            indices = np.random.choice(x_train.shape[0], batch_size, replace=False)
        else:
            indices = np.arange(x_train.shape[0])
        batch_hr_images = []
        batch_lr_images = []
        for i in indices:
            if predictors is not None:
                params = dict(predictors=predictors[i])
            else:
                params = dict()

            hr_array, lr_array = create_pair_hr_lr(
                x_train[i],
                scale=scale, 
                topography=topography, 
                landocean=landocean, 
                patch_size=patch_size, 
                model=model, 
                interpolation=interpolation,
                **params)
            batch_lr_images.append(lr_array)
            batch_hr_images.append(hr_array)

        batch_lr_images = np.asarray(batch_lr_images)
        batch_hr_images = np.asarray(batch_hr_images) 
        return batch_hr_images, batch_lr_images
    
    else:
        if shuffle:
            rangevec = np.arange(time_window, x_train.shape[0])
            indices = np.random.choice(rangevec, batch_size, replace=False)
        else:
            indices = np.arange(time_window, x_train.shape[0])
        batch_hr_images = []
        batch_lr_images = []
        batch_static_images = []
        for i in indices:
            if predictors is not None:
                params = dict(predictors=predictors[i-time_window: i])
            else:
                params = dict()

            res = create_pair_temp_hr_lr(
                x_train[i-time_window: i],
                scale=scale, 
                topography=topography, 
                landocean=landocean, 
                patch_size=patch_size, 
                model=model, 
                interpolation=interpolation,
                **params)

            if topography is not None or landocean is not None:
                hr_array, lr_array, static_array = res 
                batch_lr_images.append(lr_array)
                batch_hr_images.append(hr_array)
                batch_static_images.append(static_array)
            else:
                hr_array, lr_array = res 
                batch_lr_images.append(lr_array)
                batch_hr_images.append(hr_array)

        batch_lr_images = np.asarray(batch_lr_images)
        batch_hr_images = np.asarray(batch_hr_images) 
        if topography is not None or landocean is not None:
            batch_static_images = np.asarray(batch_static_images)
            return batch_hr_images, batch_lr_images, batch_static_images
        else:
            return batch_hr_images, batch_lr_images


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
        scale=4, 
        batch_size=32, 
        patch_size=None,
        time_window=None,
        topography=None, 
        landocean=None, 
        predictors=None,
        model='resnet_spc', 
        interpolation='bicubic',
        repeat=None,
        ):
        """
        Parameters
        ----------
        model : {'resnet_spc', 'resnet_bi', 'resnet_rc'}
            Name of the model architecture.
        predictors : list of ndarray 
            List of predictor ndarrays.

        TO-DO
        -----
        * instead of the in-memory array, we could input the path and load the 
        netcdf files lazily or memmap a numpy array
        """
        self.array = array
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
        batch_hr_images = []
        batch_lr_images = []
        batch_static_input = []

        if self.time_window is None:
            for i in self.batch_rand_idx:   
                # creating a single ndarrays concatenating list of ndarray variables along the last dimension 
                if self.predictors is not None:
                    array_predictors = np.concatenate(self.predictors, axis=-1)
                    params = dict(predictors=array_predictors[i])
                else:
                    params = {}

                res = create_pair_hr_lr(
                    array=self.array[i],
                    scale=self.scale, 
                    patch_size=self.patch_size, 
                    topography=self.topography, 
                    landocean=self.landocean, 
                    model=self.model,
                    interpolation=self.interpolation,
                    **params)
                hr_array, lr_array = res
                batch_lr_images.append(lr_array)
                batch_hr_images.append(hr_array)
            batch_lr_images = np.asarray(batch_lr_images)
            batch_hr_images = np.asarray(batch_hr_images) 
            return [batch_lr_images], [batch_hr_images]
        
        # batch of samples with a temporal dimension
        else:
            for i in self.batch_rand_idx: 
                # creating a single ndarrays concatenating list of ndarray variables along the last dimension 
                if self.predictors is not None:
                    array_predictors = np.concatenate(self.predictors, axis=-1)
                    params = dict(predictors=array_predictors[i:i+self.time_window])
                else:
                    params = {}

                res = create_pair_temp_hr_lr(
                    array=self.array[i:i+self.time_window],
                    scale=self.scale, 
                    patch_size=self.patch_size, 
                    topography=self.topography, 
                    landocean=self.landocean, 
                    model=self.model,
                    interpolation=self.interpolation,
                    **params)
                if self.topography is not None or self.landocean is not None:
                    hr_array, lr_array, static_array = res
                    batch_lr_images.append(lr_array)
                    batch_hr_images.append(hr_array)
                    batch_static_input.append(static_array)
                else:
                    hr_array, lr_array = res
                    batch_lr_images.append(lr_array)
                    batch_hr_images.append(hr_array)
            batch_lr_images = np.asarray(batch_lr_images)
            batch_hr_images = np.asarray(batch_hr_images) 
            if self.topography is not None or self.landocean is not None:
                batch_static_input = np.asarray(batch_static_input)
                return [batch_lr_images, batch_static_input], [batch_hr_images]
            else:
                return [batch_lr_images], [batch_hr_images]

    def on_epoch_end(self):
        """
        """
        np.random.shuffle(self.indices)
