from numpy.lib.function_base import quantile
import cv2
import numpy as np

import sys
sys.path.append('/esarchive/scratch/cgomez/pkgs/ecubevis/')
import ecubevis as ecv

from .resnet_mup import get_coords
from .utils import crop_array, reshape_array


def create_pair_hr_lr_preupsampling(
    array, 
    scale,
    patch_size, 
    topography=None, 
    landocean=None, 
    tuple_predictors=None, 
    debug=False, 
    interpolation='nearest'):
    """
    """
    if interpolation == 'nearest':
        interp = cv2.INTER_NEAREST
    elif interpolation == 'bicubic':
        interp = cv2.INTER_CUBIC
    elif interpolation == 'bilinear':
        interp = cv2.INTER_LINEAR

    hr_array = np.squeeze(array)
    hr_y, hr_x = hr_array.shape     
    lr_x = int(hr_x / scale)
    lr_y = int(hr_y / scale)          

    # whole image is downsampled and upsampled via interpolation
    lr_array_resized = cv2.resize(hr_array, (lr_x, lr_y), interpolation=interp)
    lr_array_resized = cv2.resize(lr_array_resized, (hr_x, hr_y), interpolation=interp)
    # cropping both hr_array and lr_array (same sizes)
    hr_array, crop_y, crop_x = crop_array(np.squeeze(hr_array), patch_size, yx=None, position=True)
    lr_array = crop_array(np.squeeze(lr_array_resized), patch_size, yx=(crop_y, crop_x))
    hr_array = np.expand_dims(hr_array, -1)
    lr_array = np.expand_dims(lr_array, -1)

    if tuple_predictors is not None:
        # expecting a tuple of 3D ndarrays [lat, lon, 1], in LR
        # turned into a 3d ndarray, [lat, lon, variables]
        lr_predictors = np.asarray(tuple_predictors)
        lr_predictors = np.rollaxis(np.squeeze(lr_predictors), 0, 3)
        # upsampling the lr predictors
        lr_predictors_resized = cv2.resize(lr_predictors, (hr_x, hr_y), interpolation=interp)
        # cropping predictors 
        lr_predictors_cropped = crop_array(lr_predictors_resized, patch_size, yx=(crop_y, crop_x))        
        lr_array = np.concatenate([lr_array, lr_predictors_cropped], axis=2)

    if topography is not None:
        # there is no need to downsize and upsize the topography if already given
        # in the HR image size 
        topography = crop_array(np.squeeze(topography), patch_size, yx=(crop_y, crop_x))
        lr_array = np.concatenate([lr_array, np.expand_dims(topography, -1)], axis=2)
            
    if landocean is not None:
        # there is no need to downsize and upsize the land-ocean mask if already given
        # in the HR image size
        landocean = crop_array(np.squeeze(landocean), patch_size, yx=(crop_y, crop_x))
        lr_array = np.concatenate([lr_array, np.expand_dims(landocean, -1)], axis=2)
    
    hr_array = np.asarray(hr_array, 'float32')
    lr_array = np.asarray(lr_array, 'float32')

    if debug:
        print(f'HR image: {hr_array.shape}, LR image resized {lr_array.shape}')
        print(f'Crop X,Y: {crop_x}, {crop_y}')

        ecv.plot_ndarray((array[:,:,0]), dpi=60, interactive=False)
        
        if topography is not None or landocean is not None or tuple_predictors is not None:
            lr_array_plot = np.squeeze(lr_array)[:,:,0]
        else:
            lr_array_plot = np.squeeze(lr_array)
        ecv.plot_ndarray((np.squeeze(hr_array), np.squeeze(lr_array_plot)), 
                         dpi=80, interactive=False, 
                         subplot_titles=('HR cropped image', 'LR cropped/resized image'))
        
        if topography is not None:
            ecv.plot_ndarray(topography, interactive=False, dpi=80, 
                             subplot_titles=('Topography'))
        
        if landocean is not None:
            ecv.plot_ndarray(landocean, interactive=False, dpi=80, 
                             subplot_titles=('Land Ocean mask'))

        if tuple_predictors is not None:
            ecv.plot_ndarray(np.rollaxis(lr_predictors_cropped, 2, 0), dpi=80, interactive=False, 
                             subplot_titles=('LR cropped predictors'), multichannel4d=True)

    return hr_array, lr_array


def create_pair_hr_lr(
    array, 
    scale, 
    patch_size, 
    topography=None, 
    landocean=None, 
    tuple_predictors=None, 
    debug=False, 
    interpolation='nearest'):
    """
    """
    if interpolation == 'nearest':
        interp = cv2.INTER_NEAREST
    elif interpolation == 'bicubic':
        interp = cv2.INTER_CUBIC
    elif interpolation == 'bilinear':
        interp = cv2.INTER_LINEAR

    hr_array = np.squeeze(array)
    lr_x, lr_y = int(patch_size / scale), int(patch_size / scale)  

    if tuple_predictors is not None:
        # expecting a tuple of 3D ndarrays [lat, lon, 1], in LR
        # turned into a 3d ndarray, [lat, lon, variables]
        array_predictors = np.asarray(tuple_predictors)
        array_predictors = np.rollaxis(np.squeeze(array_predictors), 0, 3)
        lr_array_predictors, crop_y, crop_x = crop_array(array_predictors, int(patch_size / 5),yx=None, position=True)

        # if scale is not 5, array predictors must be adapted to lr_x, lr_y size
        if scale != 5:
            lr_array_predictors=reshape_array(lr_array_predictors,(lr_x, lr_y),interp)
        crop_y, crop_x = int(crop_y * 5), int(crop_x * 5)
        hr_array = crop_array(np.squeeze(hr_array), patch_size, yx=(crop_y, crop_x))          
        lr_array = cv2.resize(hr_array, (lr_x, lr_y), interpolation=interp)
        hr_array = np.expand_dims(hr_array, -1)
        lr_array = np.expand_dims(lr_array, -1) 
        lr_array = np.concatenate([lr_array, lr_array_predictors], axis=2)
    else:
        # cropping and downsampling the image to get lr_array
        hr_array, crop_y, crop_x = crop_array(np.squeeze(hr_array), patch_size, yx=None, position=True)
        lr_array = cv2.resize(hr_array, (lr_x, lr_y), interpolation=interp)
        hr_array = np.expand_dims(hr_array, -1)
        lr_array = np.expand_dims(lr_array, -1)

    if topography is not None:
        topo_crop_hr = crop_array(np.squeeze(topography), patch_size, yx=(crop_y, crop_x))
        topo_crop_lr = cv2.resize(topo_crop_hr, (lr_x, lr_y), interpolation=interp)
        lr_array = np.concatenate([lr_array, np.expand_dims(topo_crop_lr, -1)], axis=2)
            
    if landocean is not None:
        landocean_crop_hr = crop_array(np.squeeze(landocean), patch_size, yx=(crop_y, crop_x))
        # integer array can only be interpolated with nearest method
        landocean_crop_lr = cv2.resize(landocean_crop_hr, (lr_x, lr_y), interpolation=cv2.INTER_NEAREST)
        lr_array = np.concatenate([lr_array, np.expand_dims(landocean_crop_lr, -1)], axis=2)
    
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
        
        if topography is not None:
            ecv.plot_ndarray((topo_crop_hr, topo_crop_lr), 
                             interactive=False, dpi=80, 
                             subplot_titles=('HR Topography', 'LR Topography'))
        
        if landocean is not None:
            ecv.plot_ndarray((landocean_crop_hr, landocean_crop_lr), 
                             interactive=False, dpi=80, 
                             subplot_titles=('HR Land Ocean mask', 'LR  Land Ocean mask'))

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
    model : {'rspc', 'rint', 'rmup'}
        rspc = ResNet-SPC, rint = ResNet-INT, rmup = ResNet-MUP
    predictors : tuple of 4D ndarray 
        Tuple of predictor ndarrays with dims [nsamples, lat, lon, 1].

    TO-DO: instead of the in-memory array, we could input the path and load the 
    netcdf files lazily or memmap a numpy array
    """
    if not model in ['rspc', 'rint', 'rmup']:
        raise ValueError('`model` not recognized')

    if model in ['rspc', 'rint']:
        if model == 'rspc':
            if not patch_size % scale == 0:
    		    raise ValueError('`patch_size` must be divisible by `scale`')
            create_sample_pair = create_pair_hr_lr
        else:
            create_sample_pair = create_pair_hr_lr_preupsampling

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

                res = create_sample_pair(
                        array=array[i], 
                        scale=scale, 
                        patch_size=patch_size, 
                        topography=topography, 
                        landocean=landocean, 
                        tuple_predictors=tuple_predictors,
                        interpolation=interpolation)
                hr_array, lr_array = res
                batch_lr_images.append(lr_array)
                batch_hr_images.append(hr_array)

            batch_lr_images = np.asarray(batch_lr_images)
            batch_hr_images = np.asarray(batch_hr_images) 
            yield [batch_lr_images], [batch_hr_images]
    
    elif model == 'rmup':
        max_scale = scale
        while True:
            rand_idx = np.random.permutation(array.shape[0])[:batch_size]
            batch_hr_images = []
            batch_lr_images = []
            rand_scale = np.random.uniform(1.0, max_scale)
            for i in rand_idx:
                if predictors is not None:
                    # we pass a tuple of 3D ndarrays [lat, lon, 1]
                    tuple_predictors = tuple([var[i] for var in predictors])
                else:
                    tuple_predictors = None

                res = create_pair_hr_lr(
                        array=array[i],
                        scale=rand_scale,
                        patch_size=patch_size,
                        topography=topography,
                        landocean=landocean,
                        tuple_predictors=tuple_predictors,
                        interpolation=interpolation)
                hr_array, lr_array = res
                batch_lr_images.append(lr_array)
                batch_hr_images.append(hr_array)

            patch_size_lr = int(patch_size / rand_scale)
            rand_scale=patch_size/patch_size_lr
            coords = get_coords((patch_size, patch_size), (patch_size_lr, patch_size_lr), rand_scale)
            batch_coords = batch_size * [coords]
            batch_lr_images = np.asarray(batch_lr_images)
            batch_hr_images = np.asarray(batch_hr_images)
            batch_coords = np.asarray(batch_coords)
            yield [batch_lr_images, batch_coords], [batch_hr_images]
