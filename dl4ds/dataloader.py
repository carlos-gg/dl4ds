from numpy.testing._private.utils import integer_repr
import tensorflow as tf
import numpy as np

import sys
sys.path.append('/esarchive/scratch/cgomez/pkgs/ecubevis/')
import ecubevis as ecv

from .utils import crop_array, resize_array


def create_pair_hr_lr(
    array, 
    scale, 
    patch_size, 
    topography=None, 
    landocean=None, 
    tuple_predictors=None, 
    model='rspc',
    debug=False, 
    interpolation='bicubic'):
    """
    Parameters
    ----------
    tuple_predictors : tuple of ndarrays, optional
        Tuple of 3D ndarrays [lat, lon, 1] corresponding to predictor variables,
        in low (target) resolution. Assumed to be in LR for r-spc. To be 
        concatenated to the LR version of `array`.
    
    """
    hr_array = np.squeeze(array)
    
    if model == 'rint':
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
    elif model == 'rspc':
        lr_x, lr_y = int(patch_size / scale), int(patch_size / scale)  

    if tuple_predictors is not None:
        # turned into a 3d ndarray, [lat, lon, variables]
        array_predictors = np.asarray(tuple_predictors)
        array_predictors = np.rollaxis(np.squeeze(array_predictors), 0, 3)

    if model == 'rint':
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
    elif model == 'rspc':
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
            # cropping and downsampling the image to get lr_array
            hr_array, crop_y, crop_x = crop_array(np.squeeze(hr_array), patch_size, yx=None, position=True)
            lr_array = resize_array(hr_array, (lr_x, lr_y), interpolation)    
            hr_array = np.expand_dims(hr_array, -1)
            lr_array = np.expand_dims(lr_array, -1)

    if topography is not None:
        topo_crop_hr = crop_array(np.squeeze(topography), patch_size, yx=(crop_y, crop_x))
        if model == 'rpsc':  # downsizing the topography
            topo_crop_lr = resize_array(topo_crop_hr, (lr_x, lr_y), interpolation)
            lr_array = np.concatenate([lr_array, np.expand_dims(topo_crop_lr, -1)], axis=2)
        elif model == 'rint':  # topography in HR 
            lr_array = np.concatenate([lr_array, np.expand_dims(topo_crop_hr, -1)], axis=2)

    if landocean is not None:
        landocean_crop_hr = crop_array(np.squeeze(landocean), patch_size, yx=(crop_y, crop_x))
        if model == 'rspc':  # downsizing the land-ocean mask
            # integer array can only be interpolated with nearest method
            landocean_crop_lr = resize_array(landocean_crop_hr, (lr_x, lr_y), interpolation='nearest')
            lr_array = np.concatenate([lr_array, np.expand_dims(landocean_crop_lr, -1)], axis=2)
        elif model == 'rint':  # lando in HR 
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
        
        if model == 'rspc':
            if topography is not None:
                ecv.plot_ndarray((topo_crop_hr, topo_crop_lr), 
                                interactive=False, dpi=80, 
                                subplot_titles=('HR Topography', 'LR Topography'))
            
            if landocean is not None:
                ecv.plot_ndarray((landocean_crop_hr, landocean_crop_lr), 
                                interactive=False, dpi=80, 
                                subplot_titles=('HR Land Ocean mask', 'LR  Land Ocean mask'))
        elif model == 'rint':
            if topography is not None:
                ecv.plot_ndarray(topography, interactive=False, dpi=80, 
                                 subplot_titles=('Topography'))
        
            if landocean is not None:
                ecv.plot_ndarray(landocean, interactive=False, dpi=80, 
                                 subplot_titles=('Land Ocean mask'))

        if tuple_predictors is not None:
            ecv.plot_ndarray(np.rollaxis(lr_array_predictors, 2, 0), dpi=80, interactive=False, 
                             subplot_titles=('LR cropped predictors'), multichannel4d=True)

    return hr_array, lr_array


class DataGenerator(tf.keras.utils.Sequence):
    """
    Sequence are a safer way to do multiprocessing. This structure guarantees 
    that the network will only train once on each sample per epoch which is not 
    the case with generators. 
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
        model='rspc', 
        interpolation='bicubic'):
        """
        Parameters
        ----------
        model : {'rspc', 'rint'}
            Name of the model architecture. rspc = ResNet-SPC, rint = ResNet-INT.
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
        self.model = model
        self.interpolation = interpolation
        self.n = array.shape[0]
        self.indices = np.random.permutation(self.n)

        if self.model not in ['rspc', 'rint']:        
            raise ValueError('`model` not recognized')

        if self.model == 'rspc':
            if not self.patch_size % self.scale == 0:
                raise ValueError('`patch_size` must be divisible by `scale`')

    def __len__(self):
        """
        Defines the number of batches per epoch.
        """
        return self.n // self.batch_size

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

