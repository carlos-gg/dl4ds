from numpy.lib.function_base import quantile
import cv2
import numpy as np

import sys
sys.path.append('/esarchive/scratch/cgomez/pkgs/ecubevis/')
import ecubevis as ecv

from .metasr import get_coords
from .utils import crop_image


def create_pair_hr_lr_preupsampling(
    array, 
    index, 
    scale,
    patch_size, 
    topography=None, 
    landocean=None, 
    debug=False, 
    interpolation='bicubic'):
    """
    """
    if interpolation == 'nearest':
        interp = cv2.INTER_NEAREST
    elif interpolation == 'bicubic':
        interp = cv2.INTER_CUBIC
    elif interpolation == 'bilinear':
        interp = cv2.INTER_LINEAR

    hr_array = np.squeeze(array[index])
    hr_y, hr_x = hr_array.shape     
    lr_x = int(hr_x / scale)
    lr_y = int(hr_y / scale)          
    lr_array_resized = cv2.resize(hr_array, (lr_x, lr_y), interpolation=interp)
    lr_array_resized = cv2.resize(lr_array_resized, (hr_x, hr_y), interpolation=interp)
    hr_array, crop_y, crop_x = crop_image(np.squeeze(hr_array), patch_size, 
                                          yx=None, position=True)
    lr_array = crop_image(np.squeeze(lr_array_resized), patch_size, 
                                     yx=(crop_y, crop_x))
       
    hr_array = hr_array[:,:, np.newaxis]
    lr_array = lr_array[:,:, np.newaxis]

    if topography is not None:
        # there is no need to downsize and upsize the topography if already given
        # in the HR image size 
        topography = crop_image(np.squeeze(topography), patch_size, yx=(crop_y, crop_x))
        lr_array = np.concatenate([lr_array, np.expand_dims(topography, -1)], axis=2)
            
    if landocean is not None:
        landocean = crop_image(np.squeeze(landocean), patch_size, yx=(crop_y, crop_x))
        lr_array = np.concatenate([lr_array, np.expand_dims(landocean, -1)], axis=2)
    
    hr_array = np.asarray(hr_array, 'float32')
    lr_array = np.asarray(lr_array, 'float32')

    if debug:
        print(f'HR image: {hr_array.shape}, LR image resized {lr_array.shape}')
        print(f'Crop X,Y: {crop_x}, {crop_y}')

        ecv.plot_ndarray((array[index,:,:,0]), dpi=60, interactive=False)
        
        ecv.plot_ndarray((np.squeeze(hr_array), np.squeeze(lr_array)), 
                         dpi=80, interactive=False, 
                         subplot_titles=('HR cropped image', 'LR cropped/resized image'))
        
        if topography is not None:
            ecv.plot_ndarray(topography, interactive=False, dpi=80, 
                             subplot_titles=('Topography'))
        
        if landocean is not None:
            ecv.plot_ndarray(landocean, interactive=False, dpi=80, 
                             subplot_titles=('Land Ocean mask'))

    return hr_array, lr_array


def data_loader_preupsampling(
    array, 
    scale,
    batch_size=32, 
    patch_size=40,
    topography=None, 
    landocean=None, 
    interpolation='nearest'):
    """
    instead of the in-memory array, we could input the path and load the netcdf 
    files lazily or memmap a numpy array
    """
    while True:
        batch_rand_idx = np.random.permutation(array.shape[0])[:batch_size]
        batch_hr_images = []
        batch_lr_images = []
        
        for i in batch_rand_idx:
            res = create_pair_hr_lr_preupsampling(
                    array=array, 
                    index=i, 
                    scale=scale,
                    patch_size=patch_size, 
                    topography=topography, 
                    landocean=landocean, 
                    interpolation=interpolation)
            batch_hr_images.append(res[0])
            batch_lr_images.append(res[1])

        batch_lr_images = np.asarray(batch_lr_images)
        batch_hr_images = np.asarray(batch_hr_images) 
        yield [batch_lr_images], [batch_hr_images]


def create_pair_hr_lr(
    array, 
    scale, 
    patch_size, 
    topography=None, 
    landocean=None, 
    array_predictors=None, 
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

    hr_array = array
    lr_x, lr_y = int(patch_size / scale), int(patch_size / scale)

    if array_predictors is None:
        hr_array, crop_y, crop_x = crop_image(np.squeeze(hr_array), patch_size, 
                                            yx=None, position=True)
        lr_array = cv2.resize(hr_array, (lr_x, lr_y), interpolation=interp)
        lr_array = lr_array[:,:, np.newaxis]
    else:
        # cropping first the LR predictors
        lr_array, crop_y, crop_x = crop_image(np.squeeze(array_predictors[:,:,0]), 
                                                         lr_x, yx=None, position=True)
        lr_array=np.expand_dims(lr_array, -1)
        if array_predictors.shape[2]>1:
            for i in range(array_predictors.shape[2]-1):
                lr_array2= crop_image(np.squeeze(array_predictors[:,:,i+1]), lr_x, yx=(crop_y, crop_x))
                lr_array=np.concatenate([lr_array, np.expand_dims(lr_array2, -1)], axis=2)

        crop_y, crop_x = int(crop_y * scale), int(crop_x * scale)
        hr_array = crop_image(np.squeeze(hr_array), patch_size, yx=(crop_y, crop_x))  
        
        lr_y, lr_x = lr_array.shape[:2]
        
        resized_to_lr = cv2.resize(hr_array, (lr_x, lr_y), interpolation=interp)
        lr_array = np.concatenate([np.expand_dims(resized_to_lr, -1),lr_array ], axis=2)
        
        hr_array=np.expand_dims(hr_array, -1) 

    if topography is not None:
        topo_crop_hr = crop_image(np.squeeze(topography), patch_size, yx=(crop_y, crop_x))
        topo_crop_lr = cv2.resize(topo_crop_hr, (lr_x, lr_y), interpolation=interp)
        lr_array = np.concatenate([lr_array, np.expand_dims(topo_crop_lr, -1)], axis=2)
            
    if landocean is not None:
        landocean_crop_hr = crop_image(np.squeeze(landocean), patch_size, yx=(crop_y, crop_x))
        landocean_crop_lr = cv2.resize(landocean_crop_hr, (lr_x, lr_y), interpolation=interp)
        lr_array = np.concatenate([lr_array, np.expand_dims(landocean_crop_lr, -1)], axis=2)
    
    hr_array = np.asarray(hr_array, 'float32')
    lr_array = np.asarray(lr_array, 'float32')

    if debug:
        print(f'HR image: {hr_array.shape}, LR image {lr_array.shape}')
        print(f'Crop X,Y: {crop_x}, {crop_y}')

        ecv.plot_ndarray((array[:,:,0]), dpi=60, interactive=False)
        
        if topography is not None or landocean is not None:
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

        if array_predictors is not None:
            for i in range(array_predictors.shape[2]):
                ecv.plot_ndarray(lr_array[:,:,i+1], 
                         dpi=80, interactive=False, 
                         subplot_titles=(f'LR cropped predictor {i + 1}'))

    return hr_array, lr_array


def data_loader(
    array, 
    scale=4, 
    batch_size=32, 
    patch_size=40,
    topography=None, 
    landocean=None, 
    model='edsr', 
    interpolation='nearest'):
    """
    instead of the in-memory array, we could input the path and load the netcdf 
    files lazily or memmap a numpy array
    """
    if not model in ['edsr', 'metasr']:
        raise ValueError('`model` not recognized')

    while True:
        batch_rand_idx = np.random.permutation(array.shape[0])[:batch_size]
        batch_hr_images = []
        batch_lr_images = []
        if model == 'metasr':
            scale = np.random.uniform(1.0, scale)
            coords = get_coords(hr_size=(patch_size, patch_size), 
                                lr_size=(int(patch_size / scale), 
                                         int(patch_size / scale)), scale=scale)
        
        for i in batch_rand_idx:
            res = create_pair_hr_lr(
                    array=array[i], 
                    scale=scale, 
                    patch_size=patch_size, 
                    topography=topography, 
                    landocean=landocean, 
                    interpolation=interpolation)
            hr_array, lr_array = res
            batch_lr_images.append(lr_array)
            batch_hr_images.append(hr_array)

        batch_lr_images = np.asarray(batch_lr_images)
        batch_hr_images = np.asarray(batch_hr_images) 
        
        if model == 'metasr':
            batch_coords = np.asarray(batch_size * [coords])
            yield [batch_lr_images, batch_coords], [batch_hr_images]
        elif model == 'edsr':     
            yield [batch_lr_images], [batch_hr_images]


def data_loader_metasr(
    array, 
    max_scale, 
    batch_size, 
    patch_size):
    """
    instead of the array, pass the path and load the netcdf files lazily
    """
    while True:
        rand_idx = np.random.permutation(array.shape[0])[:batch_size]
        batch_hr_images = []
        batch_lr_images = []
        rand_scale = np.random.uniform(1.0, max_scale)
        lr_x = int(patch_size / rand_scale)
        lr_y = int(patch_size / rand_scale)
        coords = get_coords((patch_size, patch_size), (lr_y, lr_x), rand_scale)
        batch_coords = batch_size * [coords]

        for i in rand_idx:
            hr_image = array[i]
            crop_y = np.random.randint(0, hr_image.shape[0] - patch_size - 1)
            crop_x = np.random.randint(0, hr_image.shape[1] - patch_size - 1)
            hr_image = hr_image[crop_y: crop_y + patch_size, crop_x: crop_x + patch_size]
            lr_image = cv2.resize(hr_image,(lr_x, lr_y), interpolation=cv2.INTER_CUBIC)
            hr_image = np.asarray(hr_image, 'float32')
            lr_image = np.asarray(lr_image, 'float32')[:,:, np.newaxis]
            batch_lr_images.append(lr_image)
            batch_hr_images.append(hr_image)

        batch_lr_images = np.asarray(batch_lr_images)
        batch_hr_images = np.asarray(batch_hr_images)
        batch_coords = np.asarray(batch_coords)
        yield [batch_lr_images, batch_coords], [batch_hr_images]
  
