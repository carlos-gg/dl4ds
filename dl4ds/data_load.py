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
                      model='edsr', interpolation='nearest'):
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
    hr_array, crop_y, crop_x = crop_image(np.squeeze(hr_array), patch_size, yx=None, position=True)
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
        ecv.plot_ndarray((array[index,:,:,0]), dpi=100, interactive=False)
        
        ecv.plot_ndarray((np.squeeze(hr_array[:,:,0]), np.squeeze(lr_array[:,:,0])), dpi=100, interactive=False, 
                         subplot_titles=('HR cropped image', 'LR cropped image'))
        
        ecv.plot_ndarray((topo_crop_hr, topo_crop_lr), interactive=False, dpi=100, 
                         subplot_titles=('HR Topography', 'LR Topography'))
        
        ecv.plot_ndarray((landocean_crop_hr, landocean_crop_lr), interactive=False, dpi=100, 
                         subplot_titles=('HR Land Ocean mask', 'LR  Land Ocean mask'))

        # ecv.plot_ndarray(pred_channels, interactive=False)

    if model == 'metasr':
        coords = get_coords((hr_y, hr_x), (lr_y, lr_x), scale)
        return hr_array, lr_array, coords
    elif model == 'edsr':
        return hr_array, lr_array
    
  
def data_loader(array, scale=4, batch_size=32, patch_size=40, debug=False,
                array_predictors=None, topography=None, landocean=None, 
                model='edsr', interpolation='nearest'):
    """
    instead of the in-memory array, we could input the path and load the netcdf files lazily or 
    memmap a numpy array
    """
    if not model in ['edsr', 'metasr']:
        raise ValueError('`model` not recognized')

    while True:
        rand_idx = np.random.permutation(array.shape[0])[:batch_size]
        batch_hr_images = []
        batch_lr_images = []
        if model == 'metasr':
            batch_coords = []
            scale = np.random.uniform(1.0, scale)
            if debug:
                print('Random scaling:', scale)
        
        for i in rand_idx:
            res = create_pair_hr_lr(array, i, scale, patch_size, array_predictors,
                                    topography, landocean, model=model, 
                                    interpolation=interpolation)
            if model == 'edsr':
                hr_array, lr_array = res
            if model == 'metasr':
                hr_array, lr_array, coords = res
                batch_coords.append(coords)
            batch_lr_images.append(np.asarray(lr_array, 'float32'))
            batch_hr_images.append(np.asarray(hr_array, 'float32'))

        batch_lr_images = np.asarray(batch_lr_images)
        batch_hr_images = np.asarray(batch_hr_images) 
        
        if model == 'metasr':
            batch_coords = np.asarray(batch_coords)
            yield [batch_lr_images, batch_coords], [batch_hr_images]
        elif model == 'edsr':     
            yield [batch_lr_images], [batch_hr_images]

