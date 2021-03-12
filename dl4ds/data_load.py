import cv2
import numpy as np

import sys
sys.path.append('/esarchive/scratch/cgomez/pkgs/ecubevis/')
import ecubevis as ecv


def get_coords(hr_size, lr_size, scale):
    """ 
    for METASR. move to another file
    """
    # scaling factor between the sizes of the LR and HR images
    scale_y = float(hr_size[0])/lr_size[0]
    scale_x = float(hr_size[1])/lr_size[1]
    # multi-dimensional grid of coordinates, with the size of the HR image
    coords = np.mgrid[0:hr_size[0], 0:hr_size[1]]    
    coords = coords.astype("float32")
    coords = np.transpose(coords, [1, 2, 0])  ## transposing   
    coords[:,:,0] = (coords[:,:,0]/scale_y) % 1
    coords[:,:,1] = (coords[:,:,1]/scale_x) % 1    
    coords = np.concatenate([coords, np.ones((hr_size[0], hr_size[1],1),"float32") / scale], 
                            axis=-1)   
    return coords


def create_pair_hr_lr(array, index, scale, patch_size, array_predictors=None, 
                      debug=False, model='edsr', interpolation='nearest'):
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
    crop_left = np.random.randint(0, hr_array.shape[0] - patch_size - 1)
    crop_top = np.random.randint(0, hr_array.shape[1] - patch_size - 1)
    hr_array = hr_array[crop_left: crop_left + patch_size, crop_top: crop_top + patch_size]
    hr_y, hr_x, _ = hr_array.shape
    lr_x = int(hr_x / scale)
    lr_y = int(hr_y / scale)
    lr_array = cv2.resize(hr_array, (lr_x, lr_y), interpolation=interp)
    lr_array = lr_array[:,:, np.newaxis]

    if array_predictors is not None:
        pred_channels = array_predictors[index] 
        pred_channels = cv2.resize(pred_channels, (full_hr_x, full_hr_y), interpolation=cv2.INTER_CUBIC)
        pred_channels = pred_channels[crop_left: crop_left + patch_size, crop_top: crop_top + patch_size]
        pred_channels = cv2.resize(pred_channels, (lr_x, lr_y), interpolation=interp)
        lr_array = np.concatenate([lr_array, pred_channels], axis=2)
    
    hr_array = np.asarray(hr_array, 'float32')
    lr_array = np.asarray(lr_array, 'float32')

    if debug:
        print(f'HR image: {hr_array.shape}, LR image {lr_array.shape}')
        ecv.plot_ndarray((array[index,:,:,0]), dpi=100, interactive=False)
            
        # if array_predictors is not None:
        #     nvars = array_predictors.shape[-1]
        #     f, ax = plt.subplots(1, nvars, figsize=(14, 4), dpi=150, sharey=True)
        #     for i in range(nvars):
        #         ax[i].imshow(array_predictors[0, :,:,i], origin='lower', cmap='viridis')
        #         ax[i].set_xticks([])
        #         ax[i].set_yticks([])
        #     f.subplots_adjust(wspace=0.01)
        
        ecv.plot_ndarray((np.squeeze(hr_array), np.squeeze(lr_array)), dpi=100, interactive=False, 
                         subplot_titles=('HR cropped image', 'LR cropped image'))
        
        # if array_predictors is not None:
        #     nvars = batch_lr_images.shape[-1]
        #     f, ax = plt.subplots(1, nvars, figsize=(14, 4), dpi=150, sharey=True)
        #     for i in range(nvars):
        #         ax[i].imshow(batch_lr_images[0, :,:,i], origin='lower', cmap='viridis')
        #         ax[i].set_xticks([])
        #         ax[i].set_yticks([])
        #     f.subplots_adjust(wspace=0.01) 

    if model == 'metasr':
        coords = get_coords((hr_y, hr_x), (lr_y, lr_x), scale)
        return hr_array, lr_array, coords
    elif model == 'edsr':
        return hr_array, lr_array
    
  
def data_loader(array, scale=4, batch_size=32, patch_size=40, debug=False, array_predictors=None, 
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
                                    model=model, interpolation=interpolation)
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