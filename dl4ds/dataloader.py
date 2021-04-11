from numpy.lib.function_base import quantile
import cv2
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
    mode='postupsampling',
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
    if not mode in ['preupsampling', 'postupsampling']:
        raise ValueError('`mode` not recognized')
    
    hr_array = np.squeeze(array)
    
    if mode == 'preupsampling':
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
    elif mode == 'postupsampling':
        lr_x, lr_y = int(patch_size / scale), int(patch_size / scale)  

    if tuple_predictors is not None:
        # turned into a 3d ndarray, [lat, lon, variables]
        array_predictors = np.asarray(tuple_predictors)
        array_predictors = np.rollaxis(np.squeeze(array_predictors), 0, 3)

    if mode == 'preupsampling':
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
    elif mode == 'postupsampling':
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
        if mode == 'postupsampling':  # downsizing the topography
            topo_crop_lr = resize_array(topo_crop_hr, (lr_x, lr_y), interpolation)
            lr_array = np.concatenate([lr_array, np.expand_dims(topo_crop_lr, -1)], axis=2)
        elif mode == 'preupsampling':  # topography in HR 
            lr_array = np.concatenate([lr_array, np.expand_dims(topo_crop_hr, -1)], axis=2)

    if landocean is not None:
        landocean_crop_hr = crop_array(np.squeeze(landocean), patch_size, yx=(crop_y, crop_x))
        if mode == 'postupsampling':  # downsizing the land-ocean mask
            # integer array can only be interpolated with nearest method
            landocean_crop_lr = resize_array(landocean_crop_hr, (lr_x, lr_y), interpolation='nearest')
            lr_array = np.concatenate([lr_array, np.expand_dims(landocean_crop_lr, -1)], axis=2)
        elif mode == 'preupsampling':  # lando in HR 
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
        
        if mode == 'postupsampling':
            if topography is not None:
                ecv.plot_ndarray((topo_crop_hr, topo_crop_lr), 
                                interactive=False, dpi=80, 
                                subplot_titles=('HR Topography', 'LR Topography'))
            
            if landocean is not None:
                ecv.plot_ndarray((landocean_crop_hr, landocean_crop_lr), 
                                interactive=False, dpi=80, 
                                subplot_titles=('HR Land Ocean mask', 'LR  Land Ocean mask'))
        elif mode == 'preupsampling':
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


def data_loader(
    array, 
    scale=4, 
    batch_size=32, 
    patch_size=40,
    topography=None, 
    landocean=None, 
    predictors=None,
    model='rspc', 
    interpolation='nearest'):
    """
    Parameters
    ----------
    model : {'rspc', 'rint'}
        rspc = ResNet-SPC, rint = ResNet-INT
    predictors : tuple of 4D ndarray 
        Tuple of predictor ndarrays with dims [nsamples, lat, lon, 1].

    TO-DO: instead of the in-memory array, we could input the path and load the 
    netcdf files lazily or memmap a numpy array
    """
    if model == 'rspc':
        mode='postupsampling'
    elif model == 'rint':
        mode='preupsampling'
    else:
        raise ValueError('`model` not recognized')

    if model == 'rspc':
        if not patch_size % scale == 0:
            raise ValueError('`patch_size` must be divisible by `scale`')

    while True:
        batch_rand_idx = np.random.permutation(array.shape[0])[:batch_size]
        batch_hr_images = []
        batch_lr_images = []

        for i in batch_rand_idx:   
            if predictors is not None:
                # we pass a tuple of 3D ndarrays [lat, lon, 1]
                tuple_predictors = tuple([var[i] for var in predictors])
            else:
                tuple_predictors = None

            res = create_pair_hr_lr(
                    array=array[i], 
                    scale=scale, 
                    patch_size=patch_size, 
                    topography=topography, 
                    landocean=landocean, 
                    tuple_predictors=tuple_predictors,
                    mode=mode,
                    interpolation=interpolation)
            hr_array, lr_array = res
            batch_lr_images.append(lr_array)
            batch_hr_images.append(hr_array)

        batch_lr_images = np.asarray(batch_lr_images)
        batch_hr_images = np.asarray(batch_hr_images) 
        yield [batch_lr_images], [batch_hr_images]
    

