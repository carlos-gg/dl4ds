from numpy.lib.function_base import quantile
import cv2
import numpy as np

import sys
sys.path.append('/esarchive/scratch/cgomez/pkgs/ecubevis/')
import ecubevis as ecv

from .metasr import get_coords
from .utils import crop_image


def create_pair_hr_lr(array, index, scale, patch_size, array_predictors=None, 
                      topography=None, landocean=None, debug=False, 
                      interpolation='nearest'):
    """
    """
    if interpolation == 'nearest':
        interp = cv2.INTER_NEAREST
    elif interpolation == 'bicubic':
        interp = cv2.INTER_CUBIC
    elif interpolation == 'bilinear':
        interp = cv2.INTER_LINEAR

    hr_array = array[index]                
    full_hr_y, full_hr_x, _ = hr_array.shape
    hr_array, crop_y, crop_x = crop_image(np.squeeze(hr_array), patch_size, 
                                          yx=None, position=True)
    hr_y, hr_x = hr_array.shape
    lr_x = int(hr_x / scale)
    lr_y = int(hr_y / scale)
    lr_array = cv2.resize(hr_array, (lr_x, lr_y), interpolation=interp)
    lr_array = lr_array[:,:, np.newaxis]

    if topography is not None:
        topo_crop_hr = crop_image(np.squeeze(topography), patch_size, yx=(crop_y, crop_x))
        topo_crop_lr = cv2.resize(topo_crop_hr, (lr_x, lr_y), interpolation=interp)
        lr_array = np.concatenate([lr_array, np.expand_dims(topo_crop_lr, -1)], axis=2)
            
    if landocean is not None:
        landocean_crop_hr = crop_image(np.squeeze(landocean), patch_size, yx=(crop_y, crop_x))
        landocean_crop_lr = cv2.resize(landocean_crop_hr, (lr_x, lr_y), interpolation=interp)
        lr_array = np.concatenate([lr_array, np.expand_dims(landocean_crop_lr, -1)], axis=2)

    # if array_predictors is not None:
    #     pred_channels = array_predictors[index] 
    #     pred_channels = cv2.resize(pred_channels, (full_hr_x, full_hr_y), interpolation=cv2.INTER_CUBIC)
    #     pred_channels = crop_image(pred_channels, patch_size, yx=None)
    #     pred_channels = cv2.resize(pred_channels, (lr_x, lr_y), interpolation=interp)
    #     lr_array = np.concatenate([lr_array, pred_channels], axis=2)
    
    hr_array = np.asarray(hr_array, 'float32')
    lr_array = np.asarray(lr_array, 'float32')

    if debug:
        print(f'HR image: {hr_array.shape}, LR image {lr_array.shape}')
        print(f'Crop X,Y: {crop_x}, {crop_y}')

        ecv.plot_ndarray((array[index,:,:,0]), dpi=60, interactive=False)
        
        ecv.plot_ndarray((np.squeeze(hr_array), np.squeeze(lr_array)), 
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

        # ecv.plot_ndarray(pred_channels, interactive=False)

    return hr_array, lr_array


def data_loader_metasr(array, max_scale, batch_size, patch_size):
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
  

def data_loader(array, scale=4, batch_size=32, patch_size=40,
                array_predictors=None, topography=None, landocean=None, 
                model='edsr', interpolation='nearest'):
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
            res = create_pair_hr_lr(array=array, 
                                    index=i, 
                                    scale=scale, 
                                    patch_size=patch_size, 
                                    array_predictors=array_predictors,
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

