import tensorflow as tf
import numpy as np

import sys
sys.path.append('/esarchive/scratch/cgomez/src/ecubevis/')
import ecubevis as ecv

from .utils import crop_array, resize_array, checkarg_model


def create_pair_hr_lrensemble(
    hr_array, 
    lr_array,
    scale, 
    patch_size, 
    topography=None, 
    landocean=None, 
    tuple_predictors=None, 
    model='rclstm_spc',
    debug=False, 
    interpolation='bicubic',
    downsample_hr=False,
    crop=True):
    """
    Create a pair of HR and LR square sub-patches. In this case, the LR
    corresponds to the ensembles of the seasonal forecast and therefore has 3 
    dimensions.
    """
    hr_array = np.squeeze(hr_array)  
    hr_array = np.expand_dims(hr_array, -1)
    lr_array = np.squeeze(lr_array)
    lr_array = np.expand_dims(lr_array, -1)
    static_array = None

    if topography is not None:
        if (topography.shape[0] != hr_array.shape[0] or
            topography.shape[1] != hr_array.shape[1]):
            raise ValueError('`topography` must be in HR (same as `hr_array`)')
    
    # if tuple_predictors is not None:
    #     # turned into a 3d ndarray, [lat, lon, variables]
    #     array_predictors = np.asarray(tuple_predictors)
    #     array_predictors = np.rollaxis(np.squeeze(array_predictors), 0, 3)

    if model in ["clstm_rspc", "conv3d_rspc"]:
        if crop:
            # cropping the lr array        
            lr_array, crop_y, crop_x = crop_array(lr_array, patch_size, yx=None, position=True) 
            # cropping the hr array
            wing = int(scale / 2)
            crop_y_hr = int(crop_y * scale) 
            crop_x_hr = int(crop_x * scale)
            patch_size_hr = int(patch_size * scale) 
            hr_array = crop_array(hr_array, patch_size_hr, yx=(crop_y_hr, crop_x_hr))

        if downsample_hr:
            hr_array = resize_array(hr_array, (patch_size, patch_size), interpolation=interpolation)
            hr_array = np.expand_dims(hr_array, -1)
            if topography is not None:
                topography = resize_array(topography, (patch_size, patch_size), interpolation=interpolation)
                static_array = topography
            if landocean is not None:
                landocean = resize_array(landocean, (patch_size, patch_size), interpolation='nearest')
                if static_array is not None:
                    static_array = np.concatenate([static_array, landocean], axis=-1)
                else:
                    static_array = landocean
        else:
            if topography is not None:
                if crop:
                    topography = crop_array(topography, patch_size_hr, yx=(crop_y_hr, crop_x_hr))
                topography = np.expand_dims(topography, -1)
                static_array = topography
            if landocean is not None:
                if crop:
                    landocean = crop_array(landocean, patch_size_hr, yx=(crop_y_hr, crop_x_hr))
                landocean = np.expand_dims(landocean, -1)
                if static_array is not None:
                    static_array = np.concatenate([static_array, landocean], axis=-1)
                else:
                    static_array = landocean

    hr_array = np.asarray(hr_array, 'float32')
    lr_array = np.asarray(lr_array, 'float32')
    
    if debug:
        print(f'HR image: {hr_array.shape}, LR image {lr_array.shape}')
        if crop:
            print(f'Crop X,Y in LR: {crop_x}, {crop_y}')
            print(f'Crop X,Y in HR: {crop_x_hr}, {crop_y_hr}')

        ecv.plot_ndarray(hr_array, dpi=100, interactive=False, 
                         plot_title='HR cropped image')

        ecv.plot_ndarray(tuple(lr_array), dpi=100, interactive=False, 
                         max_static_subplot_rows=25, horizontal_padding=0.02,
                         plot_title='LR cropped image', 
                         share_colorbar=True, share_dynamic_range=True)
        
        if topography is not None:
            ecv.plot_ndarray(topography, plot_title='HR Topography', dpi=100, 
                             interactive=False)
        
        if landocean is not None:
            ecv.plot_ndarray(landocean, plot_title='HR Land Ocean mask', dpi=100, 
                             interactive=False)

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
    tuple_predictors=None, 
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
    patch_size : int
        Size of the square patches to be extracted.
    topography : None or 2D ndarray, optional
        Elevation data.
    landocean : None or 2D ndarray, optional
        Binary land-ocean mask.
    tuple_predictors : tuple of ndarrays, optional
        Tuple of 3D ndarrays [lat, lon, 1] corresponding to predictor variables,
        in low (target) resolution. Assumed to be in LR for r-spc. To be 
        concatenated to the LR version of `array`.
    model : str, optional
        String with the name of the model architecture, either 'resnet_spc', 
        'resnet_int' or 'resnet_rec'.
    interpolation : str, optional
        Interpolation used when upsampling/downsampling the training samples.
        By default 'bicubic'. 
    debug : bool, optional
        Whether to show plots and debugging information.
    """
    hr_array = np.squeeze(array)
    
    if model == 'resnet_int':
        hr_y, hr_x = hr_array.shape 
        lr_x, lr_y = int(hr_x / scale), int(hr_y / scale) 
        # whole image is downsampled and upsampled via interpolation
        lr_array_resized = resize_array(hr_array, (lr_x, lr_y), interpolation)
        lr_array_resized = resize_array(lr_array_resized, (hr_x, hr_y), interpolation)
        # cropping both hr_array and lr_array (same sizes)
        hr_array, crop_y, crop_x = crop_array(np.squeeze(hr_array), patch_size, yx=None, position=True)
        lr_array = crop_array(np.squeeze(lr_array_resized), patch_size, yx=(crop_y, crop_x))
        hr_array = np.expand_dims(hr_array, -1)
        lr_array = np.expand_dims(lr_array, -1)
    elif model in ['resnet_spc', 'resnet_rec']:
        lr_x, lr_y = int(patch_size / scale), int(patch_size / scale)  

    if tuple_predictors is not None:
        # turned into a 3d ndarray, [lat, lon, variables]
        array_predictors = np.asarray(tuple_predictors)
        array_predictors = np.rollaxis(np.squeeze(array_predictors), 0, 3)

    if model == 'resnet_int':
        if tuple_predictors is not None:
            # upsampling the lr predictors
            array_predictors = resize_array(array_predictors, (hr_x, hr_y), interpolation)
            cropsize = patch_size
            # cropping predictors 
            lr_array_predictors, crop_y, crop_x = crop_array(array_predictors, cropsize,
                                                             yx=(crop_y, crop_x), 
                                                             position=True)
            # concatenating the predictors to the lr image
            lr_array = np.concatenate([lr_array, lr_array_predictors], axis=2)
    elif model in ['resnet_spc', 'resnet_rec']:
        if tuple_predictors is not None:
            cropsize = lr_x
            # cropping first the predictors 
            lr_array_predictors, crop_y, crop_x = crop_array(array_predictors, cropsize,
                                                             yx=None, position=True)
            crop_y = int(crop_y * scale)
            crop_x = int(crop_x * scale)
            hr_array = crop_array(np.squeeze(hr_array), patch_size, yx=(crop_y, crop_x))   
            lr_array = resize_array(hr_array, (lr_x, lr_y), interpolation)       
            hr_array = np.expand_dims(hr_array, -1)
            lr_array = np.expand_dims(lr_array, -1) 
            # concatenating the predictors to the lr image
            lr_array = np.concatenate([lr_array, lr_array_predictors], axis=2)
        else:
            # cropping the hr array
            hr_array, crop_y, crop_x = crop_array(hr_array, patch_size, yx=None, position=True)
            # downsampling the hr array to get lr_array
            lr_array = resize_array(hr_array, (lr_x, lr_y), interpolation)    
            hr_array = np.expand_dims(hr_array, -1)
            lr_array = np.expand_dims(lr_array, -1)

    if topography is not None:
        topo_crop_hr = crop_array(np.squeeze(topography), patch_size, yx=(crop_y, crop_x))
        if model in ['resnet_spc', 'resnet_rec']:  # downsizing the topography
            topo_crop_lr = resize_array(topo_crop_hr, (lr_x, lr_y), interpolation)
            lr_array = np.concatenate([lr_array, np.expand_dims(topo_crop_lr, -1)], axis=2)
        elif model == 'resnet_int':  # topography in HR 
            lr_array = np.concatenate([lr_array, np.expand_dims(topo_crop_hr, -1)], axis=2)

    if landocean is not None:
        landocean_crop_hr = crop_array(np.squeeze(landocean), patch_size, yx=(crop_y, crop_x))
        if model in ['resnet_spc', 'resnet_rec']:  # downsizing the land-ocean mask
            # integer array can only be interpolated with nearest method
            landocean_crop_lr = resize_array(landocean_crop_hr, (lr_x, lr_y), interpolation='nearest')
            lr_array = np.concatenate([lr_array, np.expand_dims(landocean_crop_lr, -1)], axis=2)
        elif model == 'resnet_int':  # lando in HR 
            lr_array = np.concatenate([lr_array, np.expand_dims(landocean_crop_hr, -1)], axis=2)
    
    hr_array = np.asarray(hr_array, 'float32')
    lr_array = np.asarray(lr_array, 'float32')

    if debug:
        print(f'HR image: {hr_array.shape}, LR image {lr_array.shape}')
        print(f'Crop X,Y: {crop_x}, {crop_y}')

        ecv.plot_ndarray((array[:,:,0]), dpi=60, interactive=False)
        
        if topography is not None or landocean is not None or tuple_predictors is not None:
            lr_array_plot = np.squeeze(lr_array)[:,:,0]
        else:
            lr_array_plot = np.squeeze(lr_array)
        ecv.plot_ndarray((np.squeeze(hr_array), lr_array_plot), 
                         dpi=80, interactive=False, 
                         subplot_titles=('HR cropped image', 'LR cropped image'))
        
        if model in ['resnet_spc', 'resnet_rec']:
            if topography is not None:
                ecv.plot_ndarray((topo_crop_hr, topo_crop_lr), 
                                interactive=False, dpi=80, 
                                subplot_titles=('HR Topography', 'LR Topography'))
            
            if landocean is not None:
                ecv.plot_ndarray((landocean_crop_hr, landocean_crop_lr), 
                                interactive=False, dpi=80, 
                                subplot_titles=('HR Land Ocean mask', 'LR  Land Ocean mask'))
        elif model == 'resnet_int':
            if topography is not None:
                ecv.plot_ndarray(topography, interactive=False, dpi=80, 
                                 subplot_titles=('HR Topography'))
        
            if landocean is not None:
                ecv.plot_ndarray(landocean, interactive=False, dpi=80, 
                                 subplot_titles=('HR Land Ocean mask'))

        if tuple_predictors is not None:
            ecv.plot_ndarray(np.rollaxis(lr_array_predictors, 2, 0), dpi=80, interactive=False, 
                             subplot_titles=('LR cropped predictors'), multichannel4d=True)

    return hr_array, lr_array


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
        patch_size=40,
        topography=None, 
        landocean=None, 
        predictors=None,
        model='resnet_spc', 
        interpolation='bicubic'
        ):
        """
        Parameters
        ----------
        model : {'resnet_spc', 'resnet_int', 'resnet_rec'}
            Name of the model architecture.
        predictors : tuple of 4D ndarray 
            Tuple of predictor ndarrays with dims [nsamples, lat, lon, 1].

        TO-DO
        -----
        instead of the in-memory array, we could input the path and load the 
        netcdf files lazily or memmap a numpy array
        """
        self.array = array
        self.batch_size = batch_size
        self.scale = scale
        self.patch_size = patch_size
        self.topography = topography
        self.landocean = landocean
        self.predictors = predictors
        self.model = checkarg_model(model)
        self.interpolation = interpolation
        self.n = array.shape[0]
        self.indices = np.random.permutation(self.n)

        if self.model in ['resnet_spc', 'resnet_rec']:
            if not self.patch_size % self.scale == 0:
                raise ValueError('`patch_size` must be divisible by `scale`')

    def __len__(self):
        """
        Defines the number of batches the DataGenerator can produce per epoch.
        A common practice is to set this value to n_samples / batch_size so that 
        the model sees the training samples at most once per epoch. 
        """
        n_batches = self.n // self.batch_size
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

        for i in self.batch_rand_idx:   
            if self.predictors is not None:
                # we pass a tuple of 3D ndarrays [lat, lon, 1]
                tuple_predictors = tuple([var[i] for var in self.predictors])
            else:
                tuple_predictors = None

            res = create_pair_hr_lr(
                array=self.array[i],
                scale=self.scale, 
                patch_size=self.patch_size, 
                topography=self.topography, 
                landocean=self.landocean, 
                tuple_predictors=tuple_predictors,
                model=self.model,
                interpolation=self.interpolation)
            hr_array, lr_array = res
            batch_lr_images.append(lr_array)
            batch_hr_images.append(hr_array)

        batch_lr_images = np.asarray(batch_lr_images)
        batch_hr_images = np.asarray(batch_hr_images) 
        return [batch_lr_images], [batch_hr_images]

    def on_epoch_end(self):
        """
        """
        np.random.shuffle(self.indices)


class DataGeneratorEns(tf.keras.utils.Sequence):
    """
    A sequence structure guarantees that the network will only train once on 
    each sample per epoch which is not the case with generators. 
    Every Sequence must implement the __getitem__ and the __len__ methods. If 
    you want to modify your dataset between epochs you may implement 
    on_epoch_end. The method __getitem__ should return a complete batch.

    """
    def __init__(self, 
        x_array, 
        y_array,
        scale=4, 
        batch_size=32, 
        patch_size=40,
        topography=None, 
        landocean=None, 
        predictors=None,
        model='resnet_spc', 
        interpolation='bicubic',
        downsample_hr=False,
        crop=True,
        repeat=False):
        """
        Parameters
        ----------
        model : {'rclstm_spc'}
            Name of the model architecture.
        predictors : tuple of 4D ndarray 
            Tuple of predictor ndarrays with dims [nsamples, lat, lon, 1].

        TO-DO
        -----
        - instead of the in-memory array, we could input the path and load the 
        netcdf files lazily or memmap a numpy array
        - when repeat=True, the x3 __len__ increase is hardcoded
        """
        self.x_array = x_array
        self.y_array = y_array
        self.batch_size = batch_size
        self.scale = scale
        self.patch_size = patch_size
        self.topography = topography
        self.landocean = landocean
        self.predictors = predictors
        self.model = checkarg_model(model)
        self.interpolation = interpolation
        self.downsample_hr = downsample_hr
        self.crop = crop
        self.repeat = repeat
        self.n = x_array.shape[0]
        self.indices = np.random.permutation(self.n)
        if self.repeat:
            self.indices = np.hstack([self.indices, self.indices, self.indices])

        if self.model in ['rclstm_spc']:
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
            return n_batches * 3
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
        batch_static_images = []

        for i in self.batch_rand_idx:   
            if self.predictors is not None:
                # we pass a tuple of 3D ndarrays [lat, lon, 1]
                tuple_predictors = tuple([var[i] for var in self.predictors])
            else:
                tuple_predictors = None

            res = create_pair_hr_lrensemble(
                hr_array = self.y_array[i],
                lr_array = self.x_array[i],
                scale=self.scale, 
                patch_size=self.patch_size, 
                topography=self.topography, 
                landocean=self.landocean, 
                tuple_predictors=tuple_predictors,
                model=self.model,
                interpolation=self.interpolation,
                downsample_hr = self.downsample_hr,
                crop=self.crop)
            
            if self.topography is not None or self.landocean is not None:
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
        if self.topography is not None or self.landocean is not None:
            batch_static_images = np.asarray(batch_static_images)
            return [batch_lr_images, batch_static_images], [batch_hr_images]
        else:
            return [batch_lr_images], [batch_hr_images]

    def on_epoch_end(self):
        """
        """
        np.random.shuffle(self.indices)

