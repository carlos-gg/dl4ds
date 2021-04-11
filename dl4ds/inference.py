import cv2
import os
import numpy as np

from .utils import resize_array


def predict_with_gt(
    model, 
    x_test, 
    scale, 
    topography=None, 
    landocean=None, 
    predictors=None, 
    interpolation='bicubic', 
    savepath=None):
    """

    Parameters
    ----------
    predictors : tuple of 4D ndarrays 
        Predictor variables, with dimensions [nsamples, lat, lon, 1].
    """ 
    model_architecture = model.name
    _, hr_y, hr_x, _ = x_test.shape
    lr_x = int(hr_x / scale)
    lr_y = int(hr_y / scale)
    
    n_channels = 1
    pos = {'pred':1, 'topo':1,'laoc':1}
    if predictors is not None:
        n_predictors = len(predictors)
        n_channels += n_predictors   
        pos['topo'] += n_predictors  
        pos['laoc'] += n_predictors 
    if topography is not None:
        n_channels += 1
        pos['laoc'] += 1
    if landocean is not None:
        n_channels += 1
    
    if model_architecture == 'rspc':
        if topography is not None:
            topo_interp = resize_array(topography, (lr_x, lr_y), interpolation)
        if landocean is not None:
            # integer array can only be interpolated with nearest method
            lando_interp = resize_array(landocean, (lr_x, lr_y), interpolation='nearest')
        x_test_lr = np.zeros((x_test.shape[0], lr_y, lr_x, n_channels))
    
        for i in range(x_test.shape[0]):
            x_test_lr[i, :, :, 0] = resize_array(x_test[i], (lr_x, lr_y), interpolation)
            if predictors is not None:
                # we create a tuple of 3D ndarrays [lat, lon, 1]
                tuple_predictors = tuple([var[i] for var in predictors])
                # turned into a 3d ndarray, [lat, lon, variables]
                array_predictors = np.asarray(tuple_predictors)
                array_predictors = np.rollaxis(np.squeeze(array_predictors), 0, 3)
                x_test_lr[i, :, :, pos['pred']:n_predictors+1] = array_predictors
            if topography is not None:
                x_test_lr[:, :, :, pos['topo']] = topo_interp
            if landocean is not None:
                x_test_lr[:, :, :, pos['laoc']] = lando_interp
    
        print('Downsampled x_test shape: ', x_test_lr.shape)

        x_test_pred = model.predict(x_test_lr)

    elif model_architecture == 'rint':
        x_test_lr = np.zeros((x_test.shape[0], hr_y, hr_x, n_channels))

        for i in range(x_test.shape[0]):
            # downsampling and upsampling via interpolation
            x_test_resized = resize_array(x_test[i], (lr_x, lr_y), interpolation)
            x_test_resized = resize_array(x_test_resized, (hr_x, hr_y), interpolation)
            x_test_lr[i, :, :, 0] = x_test_resized
            if predictors is not None:
                # we create a tuple of 3D ndarrays [lat, lon, 1]
                tuple_predictors = tuple([var[i] for var in predictors])
                # turned into a 3d ndarray, [lat, lon, variables]
                array_predictors = np.asarray(tuple_predictors)
                array_predictors = np.rollaxis(np.squeeze(array_predictors), 0, 3)
                array_predictors = resize_array(array_predictors, (hr_x, hr_y), interpolation)
                x_test_lr[i, :, :, pos['pred']:n_predictors+1] = array_predictors
            if topography is not None:
                x_test_lr[:, :, :, pos['topo']] = topography
            if landocean is not None:
                x_test_lr[:, :, :, pos['laoc']] = landocean
    
        print('Downsampled x_test shape: ', x_test_lr.shape)
        x_test_pred = model.predict(x_test_lr)

    if savepath is not None:
        name = os.path.join(savepath, 'x_test_pred.npy')
        np.save(name, x_test_pred.astype('float32'))
    return x_test_pred
